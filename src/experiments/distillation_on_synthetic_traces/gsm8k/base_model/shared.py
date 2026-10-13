"""Base-model CoT evaluation on gsm8k_test, using the MMLU eval defaults."""

from pathlib import Path

from transformers import AutoTokenizer

from core.datasets.gsm8k import GSM8KCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[5]
BASE_OUT_PATH = REPO_ROOT / "artifacts/distillation_on_synthetic_traces/gsm8k/base_model"


def evaluate_base_model(base_model_id: str, model_name: str) -> None:
    logger.info(f"Evaluating base model {base_model_id} with chain-of-thought prompts...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = EvaluatorConfig(
        model_path=base_model_id,
        eval_dataset=QADatasetAdapter(
            dataset=GSM8KCoTResponseDataset(
                config=QADatasetConfig(
                    path=str(REPO_ROOT / "data/source/gsm8k/gsm8k_test.parquet"),
                    dataset_id="gsm8k_test",
                ),
                tokenizer=tokenizer,
            ),
            add_thinking_start_token=False,
        ),
        out_path=str(BASE_OUT_PATH / f"{model_name}_cot"),
        generation=GenerationConfig(max_new_tokens=4096, max_batch_size=256),
    )
    results = Evaluator(config, tokenizer).evaluate()

    for result in results:
        logger.info(f"base_model: accuracy={result.accuracy:.4f} ({result.correct}/{result.total})")
        if result.num_truncated > 0:
            pct = result.num_truncated / result.total * 100
            logger.warning(
                f"base_model: {result.num_truncated}/{result.total} ({pct:.1f}%) sequences reached max_new_tokens"
            )
