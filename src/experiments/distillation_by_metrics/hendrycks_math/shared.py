from pathlib import Path

from pyarrow.parquet import ParquetFile
from transformers import AutoTokenizer

from core.complexity_estimation.complexity_estimator import BaseComplexityEstimator
from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.hendrycks_math.hendrycks_math_direct_response_dataset import HendrycksMathDirectResponseDataset
from core.datasets.hendrycks_math.hendrycks_math_reasoning_response_dataset import HendrycksMathReasoningResponseDataset
from core.datasets.merged_dataset_adapter import MergedDatasetAdapter
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.multi_checkpoint_evaluator import (
    GenerationConfig,
    MultiCheckpointEvaluator,
    MultiCheckpointEvaluatorConfig,
)
from core.training.base_trainer import PackingConfig
from core.training.lora_trainer import LoRASpecificTrainingArgs, LoRATrainingArgs, phi4_mini_lora_target_modules
from core.training.packing_budgets import packing_budget
from core.training.resampling_trainer import ModelGenerateConfig, ResamplingTrainer, ResamplingTrainerConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import add_average_column, merge_mmlu_on_question_id


def run(
    model_name: str,
    relative_out_path: str,
    train_dataset: str,
    sampler_cls: type[BaseDatasetSampler],
    save_schedule: list[int],
    max_thinking_tokens: int = 4096,
    skip_missing_checkpoints: bool = False,
    run_evaluation: bool = True,
    complexity_estimator_override: BaseComplexityEstimator | None = None,
    resampling_schedule: list[int] | None = None,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    learning_rate: float = 1e-5,
    shuffle: bool = False,
    seed: int = 42,
    dataset_cls: type[HendrycksMathDirectResponseDataset] = HendrycksMathReasoningResponseDataset,
):
    repo_root = Path(__file__).resolve().parents[4]
    model_path = repo_root / "artifacts/base_models_v0" / model_name
    out_path = repo_root / "artifacts/distillation_by_metrics/hendrycks_math" / relative_out_path
    teacher_entropy_path = out_path / "teacher_entropy.parquet"
    train_path = repo_root / "data/out/distillation" / f"{train_dataset}.parquet"
    train_dataset_adapter = get_merged_adapter_with_data_mix(
        sampler_cls,
        dataset_size=ParquetFile(train_path).metadata.num_rows,
        dataset_cls=dataset_cls,
    )

    # Match the proxy estimates to this exact trace pool, including augmented questions.
    # The estimation entry points already write entropy normalized by log(vocab size).
    merge_mmlu_on_question_id(
        main_path=train_path,
        extra_paths=[
            repo_root / "data/out/single_token_entropy_normalized" / f"{train_dataset}_llama_70b.parquet",
            repo_root / "data/out/single_token_entropy_normalized" / f"{train_dataset}_qwen_72b.parquet",
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

    tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix(), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)
    train_dataset_adapter.override_tokenizer(tokenizer)

    lora_training_args = lora_training_args or LoRASpecificTrainingArgs()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    trainer = ResamplingTrainer(
        config=ResamplingTrainerConfig(
            training_args=LoRATrainingArgs(
                num_train_epochs=save_schedule[-1],
                per_device_train_batch_size=1,
                learning_rate=learning_rate,
                seed=seed,
                data_seed=seed,
            ),
            lora_training_args=lora_training_args,
            packing=PackingConfig(budget=packing_budget(model_name)),
            save_schedule=save_schedule,
            resampling_schedule=resampling_schedule,
            shuffle=shuffle,
            out_path=out_path.as_posix(),
            model_id=model_path.as_posix(),
            train_dataset=train_dataset_adapter,
            complexity_evaluation_dataset=QADatasetAdapter(
                dataset=HendrycksMathDirectResponseDataset(
                    config=QADatasetConfig(
                        path=teacher_entropy_path.as_posix(),
                        dataset_id="hendrycks_math_teacher_entropy",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            complexity_estimator=complexity_estimator_override or SingleTokenEntropyWithRandomEstimator(),
            complexity_estimation_runner_generation_config=ModelGenerateConfig(max_new_tokens=1),
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

    if not run_evaluation:
        return

    eval_dataset_id = "hendrycks_math_test"
    summary_filename = "summary_reasoning_evals.json"
    if max_thinking_tokens != 8192:
        eval_dataset_id = f"hendrycks_math_test_cap{max_thinking_tokens}"
        summary_filename = f"summary_reasoning_evals_cap{max_thinking_tokens}.json"

    evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path.as_posix(),
            eval_dataset=QADatasetAdapter(
                dataset=dataset_cls(
                    config=QADatasetConfig(
                        path=(repo_root / "data/source/hendrycks_math/hendrycks_math_test.parquet").as_posix(),
                        dataset_id=eval_dataset_id,
                    ),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=True,
            ),
            generation=GenerationConfig(
                max_new_tokens=max_thinking_tokens + 30,
                max_thinking_tokens=max_thinking_tokens,
                max_batch_size=256,
            ),
            summary_filename=summary_filename,
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    evaluator.evaluate_all()


def get_merged_adapter_with_data_mix(
    sampler_cls: type[BaseDatasetSampler],
    dataset_size: int,
    dataset_cls: type[HendrycksMathDirectResponseDataset] = HendrycksMathReasoningResponseDataset,
) -> MergedDatasetAdapter:
    reasoning_size = dataset_size // 10
    direct_size = reasoning_size // 4
    # Packed resampling needs complete effective batches of 64 documents. Keep
    # the reasoning budget at 10% and adjust only the smaller answer mix (<=32 rows).
    total_size = ((reasoning_size + direct_size + 32) // 64) * 64
    direct_size = total_size - reasoning_size
    if reasoning_size < 1 or direct_size < 1:
        raise ValueError("Training pool is too small for a 10% reasoning sample with packed answer mixing")

    return MergedDatasetAdapter(
        [
            CausalDatasetAdapter(
                dataset=dataset_cls(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="train_hendrycks_math_reasoning",
                    ),
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_cls(BaseDatasetSamplerConfig(top_k=reasoning_size)),
            ),
            # Ground-truth direct answers keep first-token entropy measurable each epoch.
            CausalDatasetAdapter(
                dataset=HendrycksMathDirectResponseDataset(
                    config=QADatasetConfig(
                        path="should be overridden by SetResamplingPathCallback",
                        dataset_id="train_hendrycks_math_direct_response",
                    ),
                    tokenizer=None,  # type: ignore
                ),
                dataset_sampler=sampler_cls(BaseDatasetSamplerConfig(top_k=direct_size)),
            ),
        ]
    )
