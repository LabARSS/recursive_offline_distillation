from pathlib import Path

from transformers import AutoTokenizer

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
from core.training.base_trainer import PackingConfig
from core.training.lora_trainer import (
    LoRASpecificTrainingArgs,
    LoRASpecificTrainingArgsLarge,
    LoRATrainer,
    LoRATrainerConfig,
    LoRATrainingArgs,
    phi4_mini_lora_target_modules,
)
from core.training.packing_budgets import packing_budget
from core.training.thinking_tokens import setup_thinking_tokens
from experiments.distillation_on_synthetic_traces.gpqa.base_model.shared import self_distill_dataset_name

REPO_ROOT = Path(__file__).resolve().parents[4]


def run(
    model_name: str,
    relative_out_path: str,
    train_dataset: str,
    save_schedule: list[int],
    skip_missing_checkpoints: bool = False,
    run_evaluation: bool = True,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    data_mix: bool = False,
    self_distillation: bool = False,
):
    model_path = str(REPO_ROOT / "artifacts/base_models_v0" / model_name)
    out_path = str(REPO_ROOT / "artifacts/distillation_on_synthetic_traces/gpqa" / relative_out_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    lora_training_args = lora_training_args or LoRASpecificTrainingArgsLarge()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    train_dataset_path = str(REPO_ROOT / "data/out/splits/random/gpqa" / f"{train_dataset}.parquet")
    train_dataset_adapters = [
        CausalDatasetAdapter(
            dataset=GPQAReasoningResponseDataset(
                config=QADatasetConfig(path=train_dataset_path, dataset_id=f"gpqa_{train_dataset}"),
                tokenizer=tokenizer,
            )
        )
    ]
    if self_distillation:
        # Add the base model's own correct CoT replies so training keeps the answers it already gets right.
        self_distill_dataset = self_distill_dataset_name(model_name)
        train_dataset_adapters.append(
            CausalDatasetAdapter(
                dataset=GPQAReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=str(REPO_ROOT / "data/out/splits/random/gpqa" / f"{self_distill_dataset}.parquet"),
                        dataset_id=f"gpqa_{self_distill_dataset}",
                    ),
                    tokenizer=tokenizer,
                )
            )
        )
    if data_mix:
        # Mix in single-token answers for the same questions (no sampling, full set every epoch).
        train_dataset_adapters.append(
            CausalDatasetAdapter(
                dataset=GPQASingleTokenResponseDataset(
                    config=QADatasetConfig(path=train_dataset_path, dataset_id=f"gpqa_{train_dataset}_single_token"),
                    tokenizer=tokenizer,
                )
            )
        )
    train_dataset_adapter = (
        train_dataset_adapters[0] if len(train_dataset_adapters) == 1 else MergedDatasetAdapter(train_dataset_adapters)
    )

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=out_path,
            model_id=model_path,
            train_dataset=train_dataset_adapter,
            training_args=LoRATrainingArgs(num_train_epochs=save_schedule[-1], per_device_train_batch_size=1),
            lora_training_args=lora_training_args,
            packing=PackingConfig(budget=packing_budget(model_name)),
            save_schedule=save_schedule,
        ),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.unload()

    if not run_evaluation:
        return

    for max_thinking_tokens in [4096]:
        eval_dataset_id = f"gpqa_random_test_cap{max_thinking_tokens}"
        summary_filename = f"summary_reasoning_evals_cap{max_thinking_tokens}.json"

        evaluator = MultiCheckpointEvaluator(
            config=MultiCheckpointEvaluatorConfig(
                checkpoints_dir=out_path,
                eval_dataset=QADatasetAdapter(
                    dataset=GPQAReasoningResponseDataset(
                        config=QADatasetConfig(
                            path=str(REPO_ROOT / "data/out/splits/random/gpqa/test.parquet"),
                            dataset_id=eval_dataset_id,
                        ),
                        tokenizer=tokenizer,
                    ),
                    add_thinking_start_token=True,
                ),
                generation=GenerationConfig(
                    max_new_tokens=max_thinking_tokens + 10,
                    max_thinking_tokens=max_thinking_tokens,
                    max_batch_size=256,
                ),
                summary_filename=summary_filename,
                skip_missing_checkpoints=skip_missing_checkpoints,
            ),
            tokenizer=tokenizer,
        )
        evaluator.evaluate_all()
