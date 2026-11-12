"""Audit MATH initialization comparisons without models or network access.

Run: PYTHONPATH=src .venv/bin/python src/analysis/hendrycks_math_init_audit.py
Saved evaluations are read only. Diagnostics go under analysis/init_audit/.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path

import pandas as pd

from core.utils.cot_answers import (
    clean_assistant_response,
    extract_math_answer,
    verify_math_answer,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/distillation_on_synthetic_traces/hendrycks_math"
MIX = "reasoning_mix_under10k_omi2_aug_self"
OUT = ARTIFACTS / "analysis/init_audit"


@cache
def grade(gold: str, prediction: str) -> bool:
    return verify_math_answer(gold, prediction)


def last_box(text: str) -> str:
    """Diagnostic only: a box in reasoning need not be the final answer."""
    start = text.rfind(r"\boxed{")
    if start < 0:
        return ""
    start += len(r"\boxed{")
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        if depth == 0:
            return text[start:end].strip()
    return ""


def reparse(response: str, is_base: bool) -> str:
    # Chat-end cleanup is shared with both production dataset implementations.
    # These artifacts need no other tokenizer-specific control-token cleanup.
    if is_base:
        return extract_math_answer(response, ())
    text = clean_assistant_response(response, ())
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").replace("</think>", "").strip()


def paired(left: pd.Series, right: pd.Series) -> dict:
    right = right.reindex(left.index)
    assert right.notna().all()
    delta = right.astype(int) - left.astype(int)
    se = delta.std(ddof=1) / math.sqrt(len(delta))
    return {
        "n": len(delta),
        "left_correct": int(left.sum()),
        "right_correct": int(right.sum()),
        "gained": int((delta == 1).sum()),
        "lost": int((delta == -1).sum()),
        "delta_pp": 100 * float(delta.mean()),
        "paired_approx_95ci_pp": [
            100 * float(delta.mean() - 1.96 * se),
            100 * float(delta.mean() + 1.96 * se),
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet")
    test["question_id"] = test.question_id.astype(str)
    test = test.set_index("question_id")
    assert len(test) == 5000 and test.index.is_unique

    paths = sorted(ARTIFACTS.glob("base_model/*/hendrycks_math_test/responses.parquet"))
    paths += sorted((ARTIFACTS / MIX).rglob("responses.parquet"))
    metrics, frames, discrepancies, examples = [], {}, [], []
    for path in paths:
        label = str(path.parent.relative_to(ARTIFACTS))
        is_base = label.startswith("base_model/")
        frame = pd.read_parquet(path).set_index("row_id")
        frame.index = frame.index.astype(str)
        assert frame.index.is_unique and set(frame.index) == set(test.index), path
        frame = frame.reindex(test.index).join(test)
        saved = json.loads(path.with_name("results.json").read_text())
        assert saved["total"] == len(frame)
        assert saved["correct"] == int(frame.is_correct.sum())
        assert saved["num_truncated"] == int(frame.is_truncated.sum())
        assert saved["accuracy"] == float(frame.is_correct.mean())

        reparsed = frame.response.map(lambda response: reparse(response, is_base))
        parse_changed = reparsed.ne(frame.parsed_answer.fillna(""))
        values, boxes = [], []
        for i, row in enumerate(frame.itertuples()):
            values.append(grade(str(row.answer).strip(), str(row.parsed_answer or "").strip()))
            # Recovery diagnostics apply only to answers marked incorrect.
            text = clean_assistant_response(row.response, ())
            if "</think>" in text:
                text = text.split("</think>", 1)[0]
            boxes.append(False if row.is_correct else grade(str(row.answer).strip(), last_box(text)))
            if (i + 1) % 1000 == 0:
                print(f"{label}: {i + 1}/{len(frame)}", flush=True)
        frame["regraded_correct"] = values
        frame["reasoning_box_correct"] = boxes
        frame["reparsed_answer"] = reparsed

        empty = frame.parsed_answer.fillna("").eq("")
        changed = frame.is_correct.ne(frame.regraded_correct)
        no_cap = ~frame.is_truncated & ~frame.is_thinking_budget_exhausted
        metric = {
            "label": label,
            "model_path": saved["model_path"],
            "n": len(frame),
            "saved_correct": int(frame.is_correct.sum()),
            "regraded_correct": int(frame.regraded_correct.sum()),
            "saved_true_regraded_false": int((frame.is_correct & ~frame.regraded_correct).sum()),
            "saved_false_regraded_true": int((~frame.is_correct & frame.regraded_correct).sum()),
            "parse_disagreements": int(parse_changed.sum()),
            "empty_answer": int(empty.sum()),
            "empty_answer_untruncated": int((empty & ~frame.is_truncated).sum()),
            "truncated": int(frame.is_truncated.sum()),
            "thinking_cap": int(frame.is_thinking_budget_exhausted.sum()),
            "no_thinking_end": int((~frame.response.str.contains("</think>", regex=False)).sum()),
            "mean_response_chars": float(frame.response.str.len().mean()),
            "no_cap_n": int(no_cap.sum()),
            "no_cap_correct": int(frame.loc[no_cap, "is_correct"].sum()),
            "incorrect_with_correct_reasoning_box": int(frame.reasoning_box_correct.sum()),
            "empty_with_correct_reasoning_box": int((empty & frame.reasoning_box_correct).sum()),
        }
        metrics.append(metric)
        frames[label] = frame
        for row_id, row in frame.loc[changed | parse_changed].iterrows():
            discrepancies.append({"label": label, "row_id": row_id, **row.to_dict()})
        print(json.dumps(metric), flush=True)

    comparisons = {}
    first = {}
    for model, checkpoint in [("qwen_3b", 1794), ("llama_3b", 1582)]:
        base = frames[f"base_model/{model}_cot/hendrycks_math_test"]
        trained = frames[f"{MIX}/{model}_lora_large/checkpoint-{checkpoint}/evals/hendrycks_math_test_cap4096"]
        first[model] = trained
        no_cap = ~base.is_truncated & ~trained.is_truncated & ~trained.is_thinking_budget_exhausted
        comparisons[model] = {
            "saved": paired(base.is_correct, trained.is_correct),
            "regraded": paired(base.regraded_correct, trained.regraded_correct),
            "both_no_cap": paired(base.loc[no_cap, "is_correct"], trained.loc[no_cap, "is_correct"]),
            "identical_response_count": int(base.response.eq(trained.response).sum()),
            "base_empty_subset": paired(
                base.loc[base.parsed_answer.fillna("").eq(""), "is_correct"],
                trained.loc[base.parsed_answer.fillna("").eq(""), "is_correct"],
            ),
        }
        # Concrete failures that cannot be attributed to token-limit censoring.
        lost = base.is_correct & ~trained.is_correct & no_cap
        for row_id in lost[lost].index[:5]:
            a, b = base.loc[row_id], trained.loc[row_id]
            examples.append(
                f"## {model}: test question {row_id}\n\n{a.question}\n\n"
                f"Gold: `{a.answer}`. Original: `{a.parsed_answer}`. Trained: `{b.parsed_answer}`.\n\n"
                f"Original response:\n\n```text\n{a.response}\n```\n\n"
                f"Trained response:\n\n```text\n{b.response}\n```\n"
            )

    qwen, llama = first["qwen_3b"], first["llama_3b"]
    comparisons["trained_qwen_vs_llama"] = {
        "saved": paired(llama.is_correct, qwen.is_correct),
        "both_correct": int((qwen.is_correct & llama.is_correct).sum()),
        "identical_response_count": int(qwen.response.eq(llama.response).sum()),
    }
    result = {
        "metrics": metrics,
        "comparisons": comparisons,
        "limitations": [
            "No model execution: regrading verifies saved responses, not how they were generated.",
            "No matched MATH epoch-zero v0 reasoning evaluation exists in these artifacts.",
            "Qwen checkpoint-1794 has no local summary or trainer_state.json; epoch two is inferred from its entrypoint schedule.",
            "Box recovery is a diagnostic, not a validated replacement grader.",
            "Paired intervals describe question sampling only; they exclude seed variance and protocol confounding.",
            "Subset diagnostics condition on model outputs and are not controlled budget comparisons.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(discrepancies, columns=["label", "row_id", *frame.columns]).to_csv(
        OUT / "grading_discrepancies.csv", index=False
    )
    (OUT / "examples.md").write_text("# Uncapped failures after training\n\n" + "\n".join(examples))
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    main()
