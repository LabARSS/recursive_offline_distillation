from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.causal_dataset_adapter import CausalDatasetAdapter
from core.datasets.gpqa import GPQAReasoningResponseDataset
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

REPO_ROOT = Path(__file__).resolve().parents[4]


def run(
    model_name: str,
    relative_out_path: str,
    train_dataset: str,
    save_schedule: list[int],
    max_thinking_tokens: int = 4096,
    skip_missing_checkpoints: bool = False,
    run_evaluation: bool = True,
    lora_training_args: LoRASpecificTrainingArgs | None = None,
    learning_rate: float = 1e-4,
    eval_dataset: str = "medmcqa_test_1k",
):
    model_path = str(REPO_ROOT / "artifacts/base_models_v0" / model_name)
    out_path = str(REPO_ROOT / "artifacts/distillation_on_synthetic_traces/medmcqa" / relative_out_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setup_thinking_tokens(tokenizer)

    lora_training_args = lora_training_args or LoRASpecificTrainingArgsLarge()
    lora_training_args.train_thinking_token_embeddings = True
    if model_name == "phi4_mini":
        lora_training_args.target_modules = phi4_mini_lora_target_modules

    trainer = LoRATrainer(
        config=LoRATrainerConfig(
            out_path=out_path,
            model_id=model_path,
            train_dataset=CausalDatasetAdapter(
                dataset=GPQAReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=str(REPO_ROOT / "data/out/distillation" / f"{train_dataset}.parquet"),
                        dataset_id=f"medmcqa_{train_dataset}",
                    ),
                    tokenizer=tokenizer,
                )
            ),
            training_args=LoRATrainingArgs(
                num_train_epochs=save_schedule[-1],
                per_device_train_batch_size=1,
                learning_rate=learning_rate,
            ),
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

    evaluator = MultiCheckpointEvaluator(
        config=MultiCheckpointEvaluatorConfig(
            checkpoints_dir=out_path,
            eval_dataset=QADatasetAdapter(
                dataset=GPQAReasoningResponseDataset(
                    config=QADatasetConfig(
                        path=str(REPO_ROOT / f"data/source/medmcqa/{eval_dataset}.parquet"),
                        dataset_id=f"{eval_dataset}_cap{max_thinking_tokens}",
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
            summary_filename=f"summary_reasoning_evals_cap{max_thinking_tokens}.json",
            skip_missing_checkpoints=skip_missing_checkpoints,
        ),
        tokenizer=tokenizer,
    )
    evaluator.evaluate_all()
