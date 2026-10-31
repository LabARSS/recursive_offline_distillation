"""Audit local GPQA results and distilled chains without loading a model.

Run: .venv/bin/python src/analysis/gpqa_distillation_audit.py
Outputs go to artifacts/distillation_on_synthetic_traces/gpqa/analysis.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/distillation_on_synthetic_traces/gpqa"
SPLITS = ROOT / "data/out/splits/random/gpqa"
OUT = ARTIFACTS / "analysis"
MODELS = ("llama_3b", "phi4_mini", "qwen_3b")
END = re.compile(r"<\|(?:end|endoftext|im_end|eot_id)\|>")


def clean(text: str) -> str:
    return END.split(str(text), maxsplit=1)[0].strip()


def frame(path: Path, index: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[index] = df[index].astype(str)
    assert df[index].is_unique, path
    return df.set_index(index)


def repeat_fraction(text: str) -> float:
    words = re.findall(r"[a-z0-9']+", str(text).lower())
    grams = list(zip(words, words[1:], words[2:], words[3:]))
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def paired(left: pd.Series, right: pd.Series) -> dict:
    right = right.reindex(left.index)
    lost = int((left & ~right).sum())
    gained = int((~left & right).sum())
    n = lost + gained
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(lost, gained) + 1)) / 2**n) if n else 1.0
    return {"retained": int((left & right).sum()), "lost": lost, "gained": gained,
        "paired_exact_p_unadjusted": p}


def summary(texts: pd.Series) -> dict:
    n = texts.fillna("").str.len()
    return {"rows": len(n), "total_chars": int(n.sum()), "median_chars": float(n.median()),
        "p90_chars": float(n.quantile(.9)), "max_chars": int(n.max()),
        "median_repeated_4gram_fraction": float(texts.map(repeat_fraction).median())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = frame(SPLITS / "test.parquet", "question_id")
    original = frame(SPLITS / "train_original.parquet", "question_id")
    teacher = frame(SPLITS / "train_corrected_answer_deepseek_v4_pro_concat.parquet", "question_id")
    before = frame(SPLITS / "train_corrected_answer_deepseek_v4_pro.parquet", "question_id")
    uncorrected = frame(SPLITS / "train_distilled_deepseek_v4_flash_extend_w_v4_pro_short_clean.parquet", "question_id")
    assert not set(test.index) & set(original.index)
    assert teacher.index.equals(before.index) and teacher.index.equals(uncorrected.index)
    assert teacher.distill_reasoning.eq(before.distill_reasoning + before.corrected_reasoning.fillna("")).all()
    assert teacher.distill_answer.str.lower().eq(teacher.answer.str.lower()).all()

    correction = teacher.corrected_reasoning.fillna("").str.strip().ne("")
    source = {"teacher": summary(teacher.distill_reasoning),
        "corrected_rows": int(correction.sum()),
        "corrected_row_character_share": float(teacher.loc[correction, "distill_reasoning"].str.len().sum() /
            teacher.distill_reasoning.str.len().sum()),
        "correction_text_characters": int(teacher.corrected_reasoning.fillna("").str.len().sum()),
        "old_answers_replaced": int(teacher.distill_answer.str.lower().ne(uncorrected.distill_answer.str.lower()).sum()),
        "exact_train_test_question_overlap": int(original.question.isin(test.question).sum()),
        "teacher_questions_excluded": len(original) - len(teacher),
        "original_reasoning_cut_at_6000": int(before.distill_reasoning.str.len().eq(6000).sum()),
        "self_distillation": {}}
    teacher["trace_chars"] = teacher.distill_reasoning.str.len()
    teacher["has_correction"] = correction
    teacher["repeated_4gram_fraction"] = teacher.distill_reasoning.map(repeat_fraction)
    teacher.to_csv(OUT / "teacher_chains.csv")
    for model in MODELS:
        own = frame(SPLITS / f"train_self_distill_{model}_cot.parquet", "question_id")
        overlap = teacher.index.intersection(own.index)
        ratio = teacher.loc[overlap, "distill_reasoning"].str.len() / own.loc[overlap, "distill_reasoning"].str.len()
        source["self_distillation"][model] = {
            **summary(own.distill_reasoning), "overlap_teacher_questions": len(overlap),
            "reasoning_character_share_when_mixed": float(own.distill_reasoning.str.len().sum() /
                (own.distill_reasoning.str.len().sum() + teacher.distill_reasoning.str.len().sum())),
            "median_same_question_teacher_to_self_length_ratio": float(ratio.median()),
            "traces_under_100_chars": int(own.distill_reasoning.str.len().lt(100).sum())}

    epochs = {}
    for p in ARTIFACTS.rglob("summary_reasoning_evals_cap4096.json"):
        for rows in json.loads(p.read_text()).values():
            for row in rows:
                epochs[(str(p.parent.relative_to(ARTIFACTS)), row["checkpoint"])] = row["epoch"]

    metrics = []
    evaluations = {}
    discrepancies = []
    for p in sorted(ARTIFACTS.rglob("responses.parquet")):
        parts = p.relative_to(ARTIFACTS).parts
        arm, model_label = parts[:2]
        model = model_label.removesuffix("_cot")
        split = parts[-2]
        df = frame(p, "row_id")
        ground = original if split == "gpqa_random_train_original" else test
        assert set(df.index) == set(ground.index), p
        df = df.join(ground)
        results = json.loads(p.with_name("results.json").read_text())
        assert int(df.is_correct.sum()) == results["correct"]
        assert int(df.is_truncated.sum()) == results["num_truncated"]
        actual = df.parsed_answer.fillna("").str.strip().str.lower().eq(df.answer.str.strip().str.lower())
        assert actual.eq(df.is_correct).all(), p
        s = df.response.fillna("")
        valid = df.parsed_answer.fillna("").str.fullmatch("[a-d]")
        cap = df.is_thinking_budget_exhausted
        closing = s.str.contains("</think>", regex=False)
        normalized = s.map(clean)
        parsed_now = normalized.str.split("</think>", n=1).map(lambda x: x[-1].strip().lower())
        # Base uses a [[letter]] extractor; only compare the reasoning extractor for adapters.
        if arm != "base_model":
            correct_now = parsed_now.eq(df.answer.str.lower())
            changed = correct_now.ne(df.is_correct)
            if changed.any():
                discrepancies.extend({"run": str(p.parent.relative_to(ARTIFACTS)), "row_id": i,
                    "saved_parsed": df.loc[i, "parsed_answer"], "current_parsed": parsed_now.loc[i],
                    "saved_correct": bool(df.loc[i, "is_correct"]), "current_correct": bool(correct_now.loc[i])}
                    for i in df.index[changed])
        cp = parts[2] if arm != "base_model" else "base"
        run = str(p.parent.relative_to(ARTIFACTS))
        metric = {"run": run, "arm": arm, "model": model, "split": split,
            "checkpoint": cp, "epoch": epochs.get((f"{arm}/{model_label}", cp), 0),
            "total": len(df), "correct": int(df.is_correct.sum()), "accuracy": float(df.is_correct.mean()),
            "valid_answers": int(valid.sum()), "invalid_answers": int((~valid).sum()),
            "correct_given_valid": float(df.loc[valid, "is_correct"].mean()) if valid.any() else None,
            "thinking_cap": int(cap.sum()), "correct_at_thinking_cap": int(df.loc[cap, "is_correct"].sum()),
            "truncated": int(df.is_truncated.sum()), "has_think_end": int(closing.sum()),
            "natural_think_end": int((closing & ~cap).sum()),
            "finished_without_think_end": int((~closing & ~df.is_truncated).sum()),
            "repetition_over_half": int(s.map(repeat_fraction).gt(.5).sum()), **summary(s)}
        evaluations[run] = df
        metrics.append(metric)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "evaluation_metrics.csv", index=False)

    comparisons = []
    chosen = []
    for (arm, model), group in metrics_df[metrics_df.arm.ne("base_model")].groupby(["arm", "model"]):
        group = group.sort_values("epoch")
        best = group.loc[group.correct.idxmax()].to_dict()
        latest = group.iloc[-1].to_dict()
        base = evaluations[f"base_model/{model}_cot/gpqa_random_test"]
        for which, metric in (("best", best), ("latest", latest)):
            comp = {"arm": arm, "model": model, "selection": which, "epoch": metric["epoch"],
                "checkpoint": metric["checkpoint"], "correct": metric["correct"], "accuracy": metric["accuracy"],
                "valid_answers": metric["valid_answers"], "thinking_cap": metric["thinking_cap"],
                **paired(base.is_correct, evaluations[metric["run"]].is_correct)}
            comparisons.append(comp)
        chosen.append(best)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)

    selected_runs = {
        "llama": "corrected_reasoning_data_mix/llama_3b/checkpoint-140/evals/gpqa_random_test_cap4096",
        "phi": "corrected_reasoning_data_mix/phi4_mini/checkpoint-210/evals/gpqa_random_test_cap4096",
        "qwen": "corrected_reasoning_data_mix_self_distill/qwen_3b/checkpoint-170/evals/gpqa_random_test_cap4096",
    }
    examples = ["# GPQA response and training-chain examples\n"]
    for label, run in selected_runs.items():
        df = evaluations[run]
        model = next(m for m in MODELS if m in run)
        base = evaluations[f"base_model/{model}_cot/gpqa_random_test"]
        joined = df.join(base[["response", "parsed_answer", "is_correct"]], rsuffix="_base")
        joined.to_csv(OUT / f"{label}_paired_answers.csv")
        examples.append(f"## {label}: {run}\n")
        for row_id in ("316", "77", "90", "360"):
            row = joined.loc[row_id]
            examples.append(f"### Test question {row_id}\n\n{row.question}\n\nOptions: {row.options}\n\n"
                f"Gold: {row.answer}; base: {row.parsed_answer_base}; student: {row.parsed_answer}.\n\n"
                f"Student:\n\n```text\n{row.response}\n```\n\nBase:\n\n```text\n{row.response_base}\n```\n")
    examples.append("## Teacher chains and correction continuations\n")
    for row_id in ("240", "118", "192", "329", "522", "7"):
        row = teacher.loc[row_id]
        examples.append(f"### Training question {row_id}\n\n{row.question}\n\nOptions: {row.options}\n\n"
            f"Gold: {row.answer}; final target: {row.distill_answer}.\n\n"
            f"Original reasoning:\n\n```text\n{before.loc[row_id, 'distill_reasoning']}\n```\n\n"
            f"Appended correction:\n\n```text\n{row.corrected_reasoning}\n```\n")
    (OUT / "examples.md").write_text("\n".join(examples))
    result = {"source": source, "base_test": [m for m in metrics if m["arm"] == "base_model" and m["split"] == "gpqa_random_test"],
        "best_and_latest_paired": comparisons, "saved_scoring_discrepancies": discrepancies,
        "response_files": len(metrics), "test_questions": len(test),
        "notes": ["Length and mixture weights are characters, not tokenizer or loss weights.",
            "Best checkpoints are selected on this test set; paired p-values are descriptive and unadjusted.",
            "Current config files cannot establish historical optimizer settings without checkpoint metadata."]}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gpqa-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"corrected_reasoning": "#c04b40", "corrected_reasoning_data_mix": "#d29735",
        "corrected_reasoning_self_distill": "#44875a", "corrected_reasoning_data_mix_self_distill": "#316da0"}
    labels = {"corrected_reasoning": "Teacher chains", "corrected_reasoning_data_mix": "Teacher + direct answers",
        "corrected_reasoning_self_distill": "Teacher + own correct chains",
        "corrected_reasoning_data_mix_self_distill": "Teacher + own chains + direct"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    for i, model in enumerate(MODELS):
        baseline = metrics_df[(metrics_df.model == model) & metrics_df.arm.eq("base_model") & metrics_df.split.eq("gpqa_random_test")].iloc[0]
        for arm, color in colors.items():
            rows = metrics_df[(metrics_df.model == model) & metrics_df.arm.eq(arm)].sort_values("epoch")
            if rows.empty:
                continue
            axes[0, i].plot(rows.epoch, 100 * rows.accuracy, "o-", color=color, label=labels[arm])
            axes[1, i].plot(rows.epoch, 100 * rows.valid_answers / rows.total, "o-", color=color)
        axes[0, i].axhline(100 * baseline.accuracy, color="black", ls="--", lw=1)
        axes[1, i].axhline(100 * baseline.valid_answers / baseline.total, color="black", ls="--", lw=1)
        axes[0, i].set(title=model, ylim=(0, 45))
        axes[1, i].set(xlabel="Epoch", ylim=(0, 105))
    axes[0, 0].set_ylabel("Correct answers (%)")
    axes[1, 0].set_ylabel("Valid option letters (%)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("GPQA: correctness and answer completion on 109 held-out questions\nDashed: original-model CoT baseline; primary runs only")
    fig.savefig(OUT / "gpqa_overview.png", dpi=180)
    fig.savefig(OUT / "gpqa_overview.pdf")
    plt.close(fig)
    print(json.dumps({"files": len(metrics), "source": source, "scoring_discrepancies": discrepancies}, indent=2))


if __name__ == "__main__":
    main()
