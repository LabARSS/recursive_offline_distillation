"""Base-model CoT evaluation on gpqa_random_test, using the MMLU eval defaults."""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from core.datasets.gpqa import GPQACoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.utils.cot_answers import clean_assistant_response
from core.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[5]
BASE_OUT_PATH = REPO_ROOT / "artifacts/distillation_on_synthetic_traces/gpqa/base_model"
SPLITS_PATH = REPO_ROOT / "data/out/splits/random/gpqa"
SELF_DISTILL_SPLIT = "train_original"


def self_distill_dataset_name(model_name: str) -> str:
    return f"train_self_distill_{model_name}_cot"


def evaluate_base_model(base_model_id: str, model_name: str, split: str = "test") -> None:
    logger.info(f"Evaluating base model {base_model_id} with chain-of-thought prompts...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = EvaluatorConfig(
        model_path=base_model_id,
        eval_dataset=QADatasetAdapter(
            dataset=GPQACoTResponseDataset(
                config=QADatasetConfig(
                    path=str(SPLITS_PATH / f"{split}.parquet"),
                    dataset_id=f"gpqa_random_{split}",
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


def export_self_distill_traces(model_name: str) -> None:
    """Turn the base model's correct CoT replies on the train split into a reasoning-trace dataset."""
    responses_path = BASE_OUT_PATH / f"{model_name}_cot" / f"gpqa_random_{SELF_DISTILL_SPLIT}" / "responses.parquet"
    responses = pd.read_parquet(responses_path)
    correct = responses[responses["is_correct"].astype(str).str.lower() == "true"]

    # Drop the final [[letter]] marker; the answer goes after </think> in GPQAReasoningResponseDataset.
    reasoning = correct["response"].map(lambda r: clean_assistant_response(r, ()))
    reasoning = [r[: r.rfind("[[")].strip() for r in reasoning]
    traces = pd.DataFrame(
        {
            "question_id": correct["row_id"].astype(str).values,
            "distill_ans_correct": True,
            "distill_reasoning": reasoning,
            "distill_answer": correct["parsed_answer"].values,
        }
    )

    train = pd.read_parquet(SPLITS_PATH / f"{SELF_DISTILL_SPLIT}.parquet")
    train["question_id"] = train["question_id"].astype(str)
    df = train.merge(traces, on="question_id", how="inner", validate="one_to_one")

    out_filename = SPLITS_PATH / f"{self_distill_dataset_name(model_name)}.parquet"
    df.to_parquet(out_filename, index=False)
    logger.info(f"{model_name}: wrote {len(df)}/{len(responses)} correct CoT traces to {out_filename}")
