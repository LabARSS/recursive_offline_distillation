from collections.abc import Callable
from pathlib import Path

from transformers import AutoTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.single_token_entropy_estimator import SingleTokenEntropyEstimator
from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.gpqa import GPQAReasoningResponseDataset
from core.datasets.gpqa.gpqa_single_token_response_dataset import GPQASingleTokenResponseDataset
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.base_trainer import AbstractDatasetAdapter
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRASpecificTrainingArgsLarge,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.resampling_trainer import ModelGenerateConfig, ResamplingTrainer, ResamplingTrainerConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import add_average_column, merge_mmlu_on_question_id

REPO_ROOT = Path(__file__).resolve().parents[4]
# Proxy entropy is estimated once on the full train sample, a superset of any trace pool built from it.
PROXY_ENTROPY_DATASET = "medmcqa_train_10k"


def run(
    model_name: str,
    relative_out_path: str,
    train_dataset: str,
    train_dataset_adapter: AbstractDatasetAdapter,
    save_schedule: list[int],
    max_thinking_tokens: int = 4096,
    skip_missing_checkpoints: bool = False,
    complexity_estimator_override: BaseComplexityEstimator | None = None,
    resampling_schedule: list[int] | None = None,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    shuffle: bool = False,
    seed: int = 42,
    eval_dataset: str = "medmcqa_test_1k",
):
    model_path = (REPO_ROOT / "artifacts/base_models_v0" / model_name).as_posix()
    out_path = REPO_ROOT / "artifacts/distillation_by_metrics/medmcqa" / relative_out_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    train_dataset_adapter.override_tokenizer(tokenizer)

    lora_training_args = lora_training_args or LoRASpecificTrainingArgsLarge()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    teacher_entropy_path = out_path / "teacher_entropy.parquet"
    merge_mmlu_on_question_id(
        main_path=REPO_ROOT / "data/out/distillation" / f"{train_dataset}.parquet",
        extra_paths=[
            REPO_ROOT / "data/out/single_token_entropy_normalized" / f"{PROXY_ENTROPY_DATASET}_llama_70b.parquet",
            REPO_ROOT / "data/out/single_token_entropy_normalized" / f"{PROXY_ENTROPY_DATASET}_qwen_72b.parquet",
        ],
        extra_columns=[
            {"entropy_value": "llama_70b_entropy_value"},
            {"entropy_value": "qwen_72b_entropy_value"},
        ],
        aggregation_function=lambda df: add_average_column(
            df, "llama_70b_entropy_value", "qwen_72b_entropy_value", "teacher_entropy"
        ),
        save_path=teacher_entropy_path,
    )

    trainer = ResamplingTrainer(
        config=ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(
                num_train_epochs=save_schedule[-1], per_device_train_batch_size=2, seed=seed, data_seed=seed
            ),
            lora_training_args=lora_training_args,
            # No packing, as in MMLU: packed resampling reorders each epoch's sample by length.
            save_schedule=save_schedule,
            resampling_schedule=resampling_schedule,
            shuffle=shuffle,
            out_path=out_path.as_posix(),
            model_id=model_path,
            train_dataset=train_dataset_adapter,
            complexity_evaluation_dataset=QADatasetAdapter(
                dataset=GPQASingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path=teacher_entropy_path.as_posix(),
                        dataset_id="medmcqa_teacher_entropy",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            complexity_estimator=complexity_estimator_override or SingleTokenEntropyEstimator(),
            complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        ),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.unload()

    evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path.as_posix(),
            eval_dataset=QADatasetAdapter(
                dataset=GPQAReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=(REPO_ROOT / f"data/source/medmcqa/{eval_dataset}.parquet").as_posix(),
                        dataset_id=f"{eval_dataset}_cap{max_thinking_tokens}",
                    ),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=True,
            ),
            generation=GenerationConfig(
                max_new_tokens=max_thinking_tokens + 10, max_thinking_tokens=max_thinking_tokens, max_batch_size=256
            ),
            summary_filename=f"summary_reasoning_evals_cap{max_thinking_tokens}.json",
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    evaluator.evaluate_all()


def get_merged_adapter_with_data_mix(sampler_cls: type[BaseDatasetSampler]) -> MergedDatasetAdapter:
    return get_merged_adapter_with_data_mix_from_factory(
        lambda top_k: sampler_cls(BaseDatasetSamplerConfig(top_k=top_k))
    )


def get_merged_adapter_with_data_mix_from_factory(
    sampler_factory: Callable[[int], BaseDatasetSampler],
) -> MergedDatasetAdapter:
    return MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=GPQAReasoningResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="train_medmcqa_reasoning",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_factory(1024),
            ),
            # Mix in a smaller single-token-answer set selected by the same policy so the model
            # keeps answering with a single letter, keeping per-epoch entropy estimation stable.
            CausalDatasetAdapter(
                dataset=GPQASingleTokenResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="train_medmcqa_single_token",
                    ),
                    # Will be overridden
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_factory(256),
            ),
        ]
    )
