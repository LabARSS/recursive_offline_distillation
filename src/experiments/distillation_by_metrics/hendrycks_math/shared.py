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
from core.training.lora_trainer import LoRASpecificTrainingArgs, LoRATrainingArgs, phi4_mini_lora_target_modules
from core.training.resampling_trainer import ModelGenerateConfig, ResamplingTrainer, ResamplingTrainerConfig
from core.training.thinking_tokens import setup_thinking_tokens
from core.utils.datasets import merge_mmlu_on_question_id


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
    redraw_each_epoch: bool = False,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    learning_rate: float = 1e-5,
    shuffle: bool = False,
    seed: int = 42,
    dataset_cls: type[HendrycksMathDirectResponseDataset] = HendrycksMathReasoningResponseDataset,
    proxy_models: tuple[str, ...] = ("llama_70b", "qwen_72b"),
    proxy_entropy_dataset: str | None = None,
    eval_dataset: str = "hendrycks_math_test",
    lr_scheduler_type: str = "constant",
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

    # Match the proxy estimates to this exact trace pool, including augmented questions,
    # unless they were estimated once on a superset keyed by the same question_id (e.g. hendrycks_math_train).
    # The estimation entry points already write entropy normalized by log(vocab size).
    # teacher_entropy is the mean normalized entropy over the proxy models.
    proxy_entropy_dataset = proxy_entropy_dataset or train_dataset
    proxy_columns = [f"{proxy}_entropy_value" for proxy in proxy_models]
    merge_mmlu_on_question_id(
        main_path=train_path,
        extra_paths=[
            repo_root / "data/out/single_token_entropy_normalized" / f"{proxy_entropy_dataset}_{proxy}.parquet"
            for proxy in proxy_models
        ],
        extra_columns=[{"entropy_value": column} for column in proxy_columns],
        aggregation_function=lambda df: df.assign(
            teacher_entropy=df[proxy_columns].astype(float).mean(axis=1, skipna=False)
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
                per_device_train_batch_size=2,
                learning_rate=learning_rate,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                data_seed=seed,
            ),
            max_failed_estimation_fraction=0.25,
            lora_training_args=lora_training_args,
            # No packing, as in MMLU: packed resampling reorders each epoch's sample by length.
            save_schedule=save_schedule,
            resampling_schedule=resampling_schedule,
            redraw_each_epoch=redraw_each_epoch,
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

    eval_suffix = eval_dataset.removeprefix("hendrycks_math_test")
    eval_dataset_id = eval_dataset
    summary_filename = f"summary_reasoning_evals{eval_suffix}.json"
    if max_thinking_tokens != 8192:
        eval_dataset_id = f"{eval_dataset}_cap{max_thinking_tokens}"
        summary_filename = f"summary_reasoning_evals{eval_suffix}_cap{max_thinking_tokens}.json"

    evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path.as_posix(),
            eval_dataset=QADatasetAdapter(
                dataset=dataset_cls(
                    config=QADatasetConfig(
                        path=(repo_root / f"data/source/hendrycks_math/{eval_dataset}.parquet").as_posix(),
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
    # As in MMLU (1,024 reasoning + 256 direct): ~10% of the pool, direct answers at a quarter of that.
    reasoning_size = dataset_size // 10
    direct_size = max(reasoning_size // 4, 384)
    if direct_size < 1:
        raise ValueError("Training pool is too small for a 10% reasoning sample with answer mixing")

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
