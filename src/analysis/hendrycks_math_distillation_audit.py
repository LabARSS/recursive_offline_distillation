"""Audit saved MATH evaluations and the exact synthetic training dataset.

Run from the repository root with PYTHONPATH=src .venv/bin/python
src/analysis/hendrycks_math_distillation_audit.py. No model or network required.
Original parquet files and evaluation results are read only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from core.utils.cot_answers import clean_assistant_response, verify_math_answer


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/distillation_on_synthetic_traces/hendrycks_math"
OUT = ARTIFACT / "analysis"
SOURCE = ROOT / "data/out/distillation"
STEM = "hendrycks_math_distilled_deepseek_v4_flash_regenerate_incorrect_w_large"
GRADE_CACHE: dict[tuple[str, str], bool] = {}


def grade(gold: str, prediction: str) -> bool:
    pair = (str(gold).strip(), str(prediction).strip())
    if pair not in GRADE_CACHE:
        GRADE_CACHE[pair] = verify_math_answer(*pair)
    return GRADE_CACHE[pair]


def regrade(frame: pd.DataFrame, prediction: str, label: str) -> pd.Series:
    values = []
    for i, (gold, pred) in enumerate(zip(frame.answer, frame[prediction].fillna(""))):
        values.append(grade(gold, pred))
        if (i + 1) % 1000 == 0:
            print(f"{label}: {i + 1}/{len(frame)} regraded", flush=True)
    return pd.Series(values, index=frame.index, dtype=bool)


def lengths(series: pd.Series) -> dict:
    n = series.fillna("").str.len()
    return {
        "mean_chars": float(n.mean()),
        "median_chars": float(n.median()),
        "p90_chars": float(n.quantile(.9)),
        "p99_chars": float(n.quantile(.99)),
        "total_chars": int(n.sum()),
    }


def last_box(response: str) -> str:
    text = clean_assistant_response(response, ())
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
        if not depth:
            return text[start:end]
    return ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet")
    test = test.set_index("question_id")
    test.index = test.index.astype(str)
    train = pd.read_parquet(SOURCE / f"{STEM}_head8192.parquet").set_index("question_id")
    full = pd.read_parquet(SOURCE / f"{STEM}.parquet").set_index("question_id")
    flash = pd.read_parquet(SOURCE / "hendrycks_math_distilled_deepseek_v4_flash.parquet")
    flash = flash.set_index("question_id")
    for frame in (train, full, flash):
        frame.index = frame.index.astype(str)
        assert frame.index.is_unique
    assert len(train) == 7499 and len(test) == 5000
    assert set(train.index) <= set(full.index)
    assert train.question.eq(full.loc[train.index, "question"]).all()

    train["regraded_correct"] = regrade(train, "distill_answer", "training source")
    flash["regraded_correct"] = regrade(flash, "distill_answer", "Flash source")
    n = train.distill_reasoning.str.len()
    cut = train.distill_reasoning.ne(full.loc[train.index, "distill_reasoning"])
    assert train.distill_reasoning.eq(full.loc[train.index, "distill_reasoning"].str[:8192]).all()
    assert train.distill_answer.eq(full.loc[train.index, "distill_answer"]).all()
    changed = full.distill_reasoning.fillna("").ne(flash.distill_reasoning.fillna(""))
    source = {
        "rows": len(train),
        "saved_correct": int(train.distill_ans_correct.sum()),
        "regraded_correct": int(train.regraded_correct.sum()),
        "saved_false_regraded_true": int((~train.distill_ans_correct & train.regraded_correct).sum()),
        "saved_true_regraded_false": int((train.distill_ans_correct & ~train.regraded_correct).sum()),
        "flash_saved_correct": int(flash.distill_ans_correct.sum()),
        "flash_regraded_correct": int(flash.regraded_correct.sum()),
        "regenerated_rows": int(changed.sum()),
        "cut_rows": int(cut.sum()),
        "cut_character_share": float(n[cut].sum() / n.sum()),
        "saved_incorrect_character_share": float(n[~train.distill_ans_correct].sum() / n.sum()),
        "regraded_incorrect_character_share": float(n[~train.regraded_correct].sum() / n.sum()),
        "uncut_all_correct": int((~cut & train.regraded_correct).sum()),
        "cut_regraded_correct": int((cut & train.regraded_correct).sum()),
        "meta_instruction_rows": int(train.distill_reasoning.str.contains(
            r"answer only|final only|final answer only|no explanation|no delimiters", case=False
        ).sum()),
        "wait_rows": int(train.distill_reasoning.str.contains(r"\bwait\b", case=False).sum()),
        "train_test_exact_question_overlap": int(train.question.isin(test.question).sum()),
        "trace_lengths": lengths(train.distill_reasoning),
        "reference_lengths": lengths(train.answer_w_steps),
        "answer_character_share": float(train.distill_answer.str.len().sum() /
            (n.sum() + train.distill_answer.str.len().sum())),
        "saved_flags_equal_exact_answer_match": bool(train.distill_ans_correct.eq(
            train.answer.str.strip().eq(train.distill_answer.str.strip())
        ).all()),
        "traces_under_half_reference_length": int((n / train.answer_w_steps.str.len() < .5).sum()),
        "single_paragraph_traces": int((~train.distill_reasoning.str.contains("\n", regex=False)).sum()),
        "median_teacher_reference_length_ratio": float((n / train.answer_w_steps.str.len()).median()),
        "multi_box_train_rows": int((train.answer_w_steps.str.count(r"\\boxed\s*\{") > 1).sum()),
        "multi_box_test_rows": int((test.answer_w_steps.str.count(r"\\boxed\s*\{") > 1).sum()),
    }
    train.assign(trace_chars=n, is_cut=cut).to_csv(OUT / "training_rows.csv", index=True)

    evaluations = {}
    evaluation_metrics = []
    for path in sorted(ARTIFACT.rglob("responses.parquet")):
        relative = str(path.relative_to(ARTIFACT))
        label = relative.removesuffix("/responses.parquet")
        frame = pd.read_parquet(path).set_index("row_id")
        frame.index = frame.index.astype(str)
        assert frame.index.is_unique and set(frame.index) == set(test.index)
        frame = frame.join(test)
        assert frame.question.notna().all()
        assert frame.is_correct.sum() == json.loads(path.with_name("results.json").read_text())["correct"]
        frame["regraded_correct"] = regrade(frame, "parsed_answer", label)
        s = frame.response.fillna("")
        metric = {
            "label": label, "total": len(frame),
            "saved_correct": int(frame.is_correct.sum()),
            "regraded_correct": int(frame.regraded_correct.sum()),
            "saved_accuracy": float(frame.is_correct.mean()),
            "regraded_accuracy": float(frame.regraded_correct.mean()),
            "empty_parsed_answer": int(frame.parsed_answer.fillna("").eq("").sum()),
            "truncated": int(frame.is_truncated.sum()),
            "thinking_budget_exhausted": int(frame.is_thinking_budget_exhausted.sum()),
            "wait_rows": int(s.str.contains(r"\bwait\b", case=False).sum()),
            "meta_instruction_rows": int(s.str.contains(
                r"answer only|final only|final answer only|no explanation|no delimiters", case=False
            ).sum()),
            **lengths(s),
        }
        if label.startswith("base_model/"):
            # Diagnostic only: accept final boxed answers when required Answer: is absent.
            missing = frame.parsed_answer.fillna("").eq("")
            salvage = [grade(row.answer, last_box(row.response)) for row in frame[missing].itertuples()]
            metric["extra_correct_final_box_fallback"] = int(sum(salvage))
        evaluations[label] = frame
        evaluation_metrics.append(metric)

    base = evaluations["base_model/llama_3b_cot/hendrycks_math_test"]
    comparisons = []
    for label, frame in evaluations.items():
        if not label.startswith("direct_reasoning_trace/llama"):
            continue
        for grading in ("is_correct", "regraded_correct"):
            b = base[grading].reindex(frame.index)
            lost = b & ~frame[grading]
            gained = ~b & frame[grading]
            comparisons.append({
                "label": label, "grading": grading,
                "lost": int(lost.sum()), "gained": int(gained.sum()),
                "lost_without_truncation": int((lost & ~frame.is_truncated).sum()),
                "lost_without_thinking_cap": int((lost & ~frame.is_thinking_budget_exhausted).sum()),
                "median_lost_response_chars": float(frame.loc[lost, "response"].str.len().median()),
                "median_base_response_chars_on_lost": float(base.loc[lost, "response"].str.len().median()),
            })

    checkpoint_changes = []
    keys = sorted((k for k in evaluations if k.startswith("direct_reasoning_trace/llama")),
        key=lambda k: int(re.search(r"checkpoint-(\d+)", k).group(1)))
    for a, b in zip(keys, keys[1:]):
        left = evaluations[a].regraded_correct
        right = evaluations[b].regraded_correct.reindex(left.index)
        checkpoint_changes.append({"from": a, "to": b,
            "lost": int((left & ~right).sum()), "gained": int((~left & right).sum())})

    result = {"source": source, "evaluations": evaluation_metrics,
        "paired_vs_base": comparisons, "checkpoint_changes": checkpoint_changes,
        "length_unit": "characters; student tokenizer is unavailable locally"}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(evaluation_metrics).to_csv(OUT / "evaluation_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)

    final = evaluations[keys[-1]]
    joined = final.join(base[["response", "parsed_answer", "is_correct", "regraded_correct"]], rsuffix="_base")
    lost = joined.regraded_correct_base & ~joined.regraded_correct
    joined.loc[lost].to_csv(OUT / "llama_lost_answers.csv", index=True)

    examples = []
    for row_id in ("2463", "1634", "832", "133", "1625", "4864"):
        row = joined.loc[row_id]
        examples.append(f"### Test question {row_id}\n\n{row.question}\n\nGold: `{row.answer}`; "
            f"base: `{row.parsed_answer_base}`; checkpoint 1690: `{row.parsed_answer}`.\n\n"
            f"Checkpoint response:\n\n```text\n{row.response}\n```\n\n"
            f"Base response:\n\n```text\n{row.response_base}\n```\n")
    examples.append("## Training traces marked incorrect but currently verified\n")
    for row_id in ("6309", "737", "55", "1855"):
        row = train.loc[row_id]
        examples.append(f"### Training question {row_id}\n\n{row.question}\n\nGold: `{row.answer}`; "
            f"teacher: `{row.distill_answer}`; saved correct: {row.distill_ans_correct}; "
            f"currently verified: {row.regraded_correct}.\n\n```text\n{row.distill_reasoning}\n```\n")
    examples.append("## Traces cut before the solution is finished\n")
    for row_id in ("4512", "2976", "2931"):
        row = train.loc[row_id]
        examples.append(f"### Training question {row_id}\n\n{row.question}\n\n"
            f"Appended teacher final answer: `{row.distill_answer}`.\n\n"
            f"Training reasoning tail:\n\n```text\n{row.distill_reasoning[-900:]}\n```\n\n"
            f"Original reasoning tail:\n\n```text\n{full.loc[row_id, 'distill_reasoning'][-900:]}\n```\n")
    current_bad = train[~train.regraded_correct]
    examples.append("## Source answers that fail the current verifier\n\n"
        "Some failures are notation or answer-form differences; verifier failure is not proof of a math error.\n")
    for row_id, row in current_bad.sample(min(20, len(current_bad)), random_state=42).iterrows():
        examples.append(f"### Training question {row_id}\n\n{row.question}\n\n"
            f"Gold: `{row.answer}`; teacher: `{row.distill_answer}`.\n\n"
            f"```text\n{full.loc[row_id, 'distill_reasoning']}\n```\n\n"
            f"Reference:\n\n```text\n{row.answer_w_steps}\n```\n")
    (OUT / "examples.md").write_text("# MATH distillation audit examples\n\n" + "\n".join(examples))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    checkpoints = [base] + [evaluations[key] for key in keys]
    axes[0].bar(["Base CoT", "338", "845", "1690"],
        [100 * frame.regraded_correct.mean() for frame in checkpoints], color="#37699a")
    axes[0].set(title="Llama MATH accuracy", ylabel="Percent correct", ylim=(0, 60))
    for i, frame in enumerate(checkpoints):
        axes[0].text(i, 100 * frame.regraded_correct.mean() + 1,
            f"{100 * frame.regraded_correct.mean():.2f}%", ha="center", fontsize=9)
    axes[1].bar(["Base CoT", "Checkpoint 1690"],
        [joined.loc[lost, "response_base"].str.len().median(),
         joined.loc[lost, "response"].str.len().median()], color=["#37699a", "#d27a32"])
    axes[1].set(title="Same 1,233 questions lost", ylabel="Median response characters")
    axes[2].bar(["Share of examples", "Share of reasoning\ncharacters"],
        [100 * cut.mean(), 100 * source["cut_character_share"]], color="#d27a32")
    axes[2].set(title="326 unfinished training traces", ylabel="Percent", ylim=(0, 35))
    fig.suptitle("MATH distillation audit: Llama 3.2 3B", fontsize=14)
    fig.savefig(OUT / "audit_overview.png", dpi=180)
    fig.savefig(OUT / "audit_overview.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
