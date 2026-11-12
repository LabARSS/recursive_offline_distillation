"""Generate one greedy CoT per MATH train question and export correct traces."""

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from core.datasets.hendrycks_math import HendrycksMathCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.utils.cot_answers import clean_assistant_response
from core.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[5]
GREEDY_DATASET_ID = "hendrycks_math_train_greedy"
OUTPUT_COLUMNS = [
    "question", "answer", "answer_w_steps", "question_id", "distill_ans_correct",
    "distill_reasoning", "distill_answer", "pass",
]


def strip_answer_line(response: str, special_tokens: list[str]) -> str:
    """Keep the full reasoning, with its final Answer line stored separately."""
    lines = clean_assistant_response(response, special_tokens).splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines[:-1]).strip()


def run_greedy_self_distillation(model_id: str, model_name: str) -> None:
    train_path = REPO_ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet"
    distill_dir = REPO_ROOT / "data/out/distillation"
    raw_path = distill_dir / f"hendrycks_math_self_distilled_{model_name}_raw"
    output_path = distill_dir / f"hendrycks_math_self_distilled_{model_name}.parquet"
    distill_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    Evaluator(
        EvaluatorConfig(
            model_path=model_id,
            eval_dataset=QADatasetAdapter(
                dataset=HendrycksMathCoTResponseDataset(
                    config=QADatasetConfig(path=str(train_path), dataset_id=GREEDY_DATASET_ID),
                    tokenizer=tokenizer,
                ),
                add_thinking_start_token=False,
            ),
            out_path=str(raw_path),
            generation=GenerationConfig(max_new_tokens=4096, max_batch_size=256, temperature=0.0),
        ),
        tokenizer,
    ).evaluate()

    responses = pd.read_parquet(raw_path / GREEDY_DATASET_ID / "responses.parquet")
    logger.info(f"{model_name} greedy: accuracy {responses.is_correct.mean():.4f}")
    correct = responses[responses.is_correct & ~responses.is_truncated].drop_duplicates("row_id").copy()
    correct["row_id"] = correct.row_id.astype(str)
    train = pd.read_parquet(train_path)
    train["question_id"] = train.question_id.astype(str)
    out = train.merge(correct, left_on="question_id", right_on="row_id", validate="one_to_one")
    out["distill_reasoning"] = out.response.map(lambda r: strip_answer_line(r, tokenizer.all_special_tokens))
    out["distill_answer"] = out.parsed_answer
    out["distill_ans_correct"] = True
    out["pass"] = GREEDY_DATASET_ID
    out = out[out.distill_reasoning.astype(bool)][OUTPUT_COLUMNS]
    out.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(out)}/{len(train)} questions with a correct greedy trace to {output_path}")
