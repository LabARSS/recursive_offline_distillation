"""Generate self-distillation traces: Llama-3.2-3B-Instruct's own correct CoT on the Hendrycks MATH train split.

Uses the same prompt and decoding as the base-model eval
(experiments/distillation_on_synthetic_traces/hendrycks_math/base_model), so the
traces are on-policy. Runs one greedy pass plus SAMPLED_PASSES sampled passes and
keeps one correct trace per question, preferring the greedy one.

Run from the repository root:
  PYTHONPATH=src python -m experiments.generate_reasoning_traces.reasoning_trace.hendrycks_math.llama_3b_self

Outputs:
  data/out/distillation/hendrycks_math_self_distilled_llama_3b_raw/  (per-pass responses)
  data/out/distillation/hendrycks_math_self_distilled_llama_3b.parquet
"""

from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from core.datasets.hendrycks_math import HendrycksMathCoTResponseDataset
from core.datasets.qa_dataset import QADatasetConfig
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.evaluation.evaluator import Evaluator, EvaluatorConfig, GenerationConfig
from core.utils.cot_answers import clean_assistant_response
from core.utils.logger import logger

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "llama_3b"
SAMPLED_PASSES = 3
SAMPLING_TEMPERATURE = 0.7

REPO_ROOT = Path(__file__).resolve().parents[5]
TRAIN_PATH = REPO_ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet"
RAW_OUT_PATH = REPO_ROOT / f"data/out/distillation/hendrycks_math_self_distilled_{MODEL_NAME}_raw"
OUTPUT_PATH = REPO_ROOT / f"data/out/distillation/hendrycks_math_self_distilled_{MODEL_NAME}.parquet"


def generate(tokenizer, dataset_ids: list[str], temperature: float) -> None:
    Evaluator(
        EvaluatorConfig(
            model_path=MODEL_ID,
            eval_dataset=[
                QADatasetAdapter(
                    dataset=HendrycksMathCoTResponseDataset(
                        config=QADatasetConfig(path=str(TRAIN_PATH), dataset_id=dataset_id),
                        tokenizer=tokenizer,
                    ),
                    add_thinking_start_token=False,
                )
                for dataset_id in dataset_ids
            ],
            out_path=str(RAW_OUT_PATH),
            generation=GenerationConfig(max_new_tokens=4096, max_batch_size=256, temperature=temperature),
        ),
        tokenizer,
    ).evaluate()


def strip_answer_line(response: str, special_tokens: list[str]) -> str:
    """Drop the final 'Answer: ...' line; the answer goes after </think> in training."""
    lines = clean_assistant_response(response, special_tokens).splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines[:-1]).strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    greedy_id = "hendrycks_math_train_greedy"
    sampled_ids = [f"hendrycks_math_train_t{SAMPLING_TEMPERATURE}_s{i}" for i in range(SAMPLED_PASSES)]
    generate(tokenizer, [greedy_id], temperature=0.0)
    generate(tokenizer, sampled_ids, temperature=SAMPLING_TEMPERATURE)

    # Greedy first so drop_duplicates keeps it whenever it is correct.
    passes = []
    for dataset_id in [greedy_id, *sampled_ids]:
        responses = pd.read_parquet(RAW_OUT_PATH / dataset_id / "responses.parquet")
        responses["pass"] = dataset_id
        logger.info(f"{dataset_id}: accuracy {responses.is_correct.mean():.4f}")
        passes.append(responses)
    responses = pd.concat(passes)
    correct = responses[responses.is_correct & ~responses.is_truncated].drop_duplicates("row_id")

    train = pd.read_parquet(TRAIN_PATH)
    out = train.merge(correct, left_on="question_id", right_on="row_id")
    out["distill_reasoning"] = out.response.map(lambda r: strip_answer_line(r, tokenizer.all_special_tokens))
    out["distill_answer"] = out.parsed_answer
    out["distill_ans_correct"] = True
    out = out[out.distill_reasoning.astype(bool)]
    out = out[
        [
            "question",
            "answer",
            "answer_w_steps",
            "question_id",
            "distill_ans_correct",
            "distill_reasoning",
            "distill_answer",
            "pass",
        ]
    ]

    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info(
        f"Saved {len(out)}/{len(train)} questions with a correct trace to {OUTPUT_PATH}; "
        f"by pass: {out['pass'].value_counts().to_dict()}"
    )


if __name__ == "__main__":
    freeze_support()
    main()
