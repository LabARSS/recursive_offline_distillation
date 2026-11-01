"""Compare new public-mixture GPQA results with original and older checkpoints.

Run from the repository root:
  .venv/bin/python src/analysis/gpqa_public_mix_results.py

Does not load a model, access the network, or change original response files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from gpqa_distillation_audit import clean, frame, paired, repeat_fraction


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/distillation_on_synthetic_traces/gpqa"
OUT = ARTIFACTS / "analysis/public_mix_2026_11_01"
SPLITS = ROOT / "data/out/splits/random/gpqa"
ARMS = {
    "public_reasoning_mix": "Dataset only",
    "public_reasoning_mix_data_mix": "+ direct answers",
    "public_reasoning_mix_self_distill": "+ self-distillation",
    "public_reasoning_mix_data_mix_self_distill": "+ direct + self",
}
MODELS = ("llama_3b", "phi4_mini", "qwen_3b")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test = frame(SPLITS / "test.parquet", "question_id")
    train = frame(SPLITS / "train_original.parquet", "question_id")
    manifest = json.loads((SPLITS / "train_reasoning_mix_public.manifest.json").read_text())
    mix_path = SPLITS / "train_reasoning_mix_public.parquet"
    mix = pd.read_parquet(mix_path)
    assert hashlib.sha256(mix_path.read_bytes()).hexdigest() == manifest["output_sha256"]
    for key in ("train", "test"):
        entry = manifest["input_files"][key]
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert len(mix) == 1484 and mix.original_question_id.nunique() == 437
    assert not mix.question.isin(test.question).any()
    assert set(mix.original_question_id) <= set(train.index)
    assert not set(mix.original_question_id) & set(test.index)

    epochs = {}
    for path in ARTIFACTS.rglob("summary_reasoning_evals_cap4096.json"):
        for rows in json.loads(path.read_text()).values():
            for row in rows:
                epochs[(str(path.parent.relative_to(ARTIFACTS)), row["checkpoint"])] = row["epoch"]

    records, frames = [], {}
    for path in sorted(ARTIFACTS.rglob("responses.parquet")):
        parts = path.relative_to(ARTIFACTS).parts
        arm, model = parts[:2]
        if arm == "base_model" and parts[-2] != "gpqa_random_test":
            continue
        model = model.removesuffix("_cot")
        df = frame(path, "row_id")
        assert set(df.index) == set(test.index)
        df = df.reindex(test.index).join(test)
        result = json.loads(path.with_name("results.json").read_text())
        assert len(df) == result["total"] == 109
        assert int(df.is_truncated.sum()) == result["num_truncated"]
        parsed = []
        for response in df.response:
            content = clean(response)
            if arm == "base_model":
                position = content.rfind("[[")
                match = re.match(r"\[\[\s*([a-d])\s*\]\]", content[position:], re.IGNORECASE) if position >= 0 else None
                parsed.append(match[1].lower() if match else "")
            else:
                # The dataset verifier removes registered special tokens from
                # the answer, including repeated thinking markers.
                answer = content.split("</think>", 1)[-1]
                parsed.append(answer.replace("<think>", "").replace("</think>", "").strip().lower())
        parsed = pd.Series(parsed, index=df.index)
        correct = parsed.eq(df.answer.str.lower())
        assert correct.eq(df.is_correct).all(), path
        assert int(correct.sum()) == result["correct"]
        valid = parsed.str.fullmatch("[a-d]")
        chars = df.response.str.len()
        run = str(path.parent.relative_to(ARTIFACTS))
        checkpoint = parts[2] if arm != "base_model" else "base"
        epoch = epochs.get((f"{arm}/{model}", checkpoint), 0)
        if arm in ARMS:
            assert epoch in (2, 5, 10, 15, 20)
        frames[run] = df
        records.append({"run": run, "arm": arm, "model": model, "epoch": epoch,
            "checkpoint": checkpoint, "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "valid": int(valid.sum()), "invalid": int((~valid).sum()),
            "thinking_cap": int(df.is_thinking_budget_exhausted.sum()), "truncated": int(df.is_truncated.sum()),
            "median_chars": float(chars.median()), "mean_chars": float(chars.mean()),
            "p90_chars": float(chars.quantile(.9)), "responses_over_10k_chars": int(chars.gt(10000).sum()),
            "repetition_over_half": int(df.response.map(repeat_fraction).gt(.5).sum())})
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "all_test_evaluations.csv", index=False)
    public = metrics[metrics.arm.isin(ARMS)].sort_values(["model", "arm", "epoch"])
    public.to_csv(OUT / "public_checkpoint_metrics.csv", index=False)

    selections, comparisons = [], []
    for (model, arm), group in public.groupby(["model", "arm"]):
        group = group.sort_values("epoch")
        for selection, record in (("best", group.loc[group.correct.idxmax()]), ("latest", group.iloc[-1])):
            row = record.to_dict()
            row["selection"] = selection
            row["mean_correct_across_saved_checkpoints"] = float(group.correct.mean())
            selections.append(row)
            base = metrics[metrics.model.eq(model) & metrics.arm.eq("base_model")].iloc[0]
            previous = metrics[metrics.model.eq(model) & metrics.arm.str.startswith("corrected_reasoning")]
            old_same_arm = previous[previous.arm.eq(arm.replace("public_reasoning_mix", "corrected_reasoning", 1))]
            references = [("original_model", base), ("previous_best_any_arm", previous.loc[previous.correct.idxmax()])]
            if not old_same_arm.empty:
                references.append(("previous_best_same_arm", old_same_arm.loc[old_same_arm.correct.idxmax()]))
            for reference, old in references:
                left, right = frames[old.run], frames[record.run]
                comparison = paired(left.is_correct, right.is_correct)
                old_valid = left.parsed_answer.str.fullmatch("[a-d]")
                new_valid = right.parsed_answer.str.fullmatch("[a-d]")
                comparisons.append({"model": model, "arm": arm, "selection": selection,
                    "new_checkpoint": record.checkpoint, "new_epoch": record.epoch,
                    "new_correct": int(record.correct), "reference": reference, "old_run": old.run,
                    "old_correct": int(old.correct), "old_valid": int(old_valid.sum()),
                    "new_valid": int(new_valid.sum()),
                    "gains_from_old_invalid_outputs": int((~left.is_correct & right.is_correct & ~old_valid).sum()),
                    **comparison})
    pd.DataFrame(selections).to_csv(OUT / "best_and_latest.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)

    missing = [{"model": model, "arm": arm} for model in MODELS for arm in ARMS
        if public[public.model.eq(model) & public.arm.eq(arm)].empty]
    source_stats = mix.groupby("source").agg(rows=("question_id", "size"),
        questions=("original_question_id", "nunique"), median_chars=("distill_reasoning", lambda s: s.str.len().median()),
        total_chars=("distill_reasoning", lambda s: int(s.str.len().sum()))).reset_index()
    source_stats["character_share"] = source_stats.total_chars / source_stats.total_chars.sum()
    summary = {"test_questions": 109, "new_evaluation_files": len(public),
        "all_test_evaluation_files_checked": len(metrics), "available_new_runs": len(public.groupby(["arm", "model"])),
        "missing_new_runs": missing, "dataset_integrity": {"rows": len(mix),
            "training_questions": mix.original_question_id.nunique(), "test_overlap": 0,
            "matches_build_manifest_hash": True},
        "source_composition": source_stats.to_dict("records"),
        "selections": selections, "paired_comparisons": comparisons,
        "caveats": ["Best checkpoints are selected on the test set; paired p-values are unadjusted for this selection.",
            "No new Phi evaluations or Llama self-distillation evaluations are present.",
            "Current scripts specify 1e-4; historical optimizer logs/weights are not available locally.",
            "Adding direct and own-chain examples changes the number of optimizer steps at the same epoch.",
            "Final-answer scoring does not establish validity of the explanations."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    chosen = {
        "llama_dataset_only": "public_reasoning_mix/llama_3b/checkpoint-250/evals/gpqa_random_test_cap4096",
        "qwen_self": "public_reasoning_mix_self_distill/qwen_3b/checkpoint-600/evals/gpqa_random_test_cap4096",
        "qwen_direct": "public_reasoning_mix_data_mix/qwen_3b/checkpoint-360/evals/gpqa_random_test_cap4096",
    }
    example_ids = {"llama_dataset_only": ("324", "360", "77"), "qwen_self": ("316", "530", "394"),
        "qwen_direct": ("316", "360", "77")}
    examples = ["# New public-mixture GPQA response examples\n\n"
        "These are diagnostic examples, not a random sample or a count of all reasoning errors.\n"]
    for label, run in chosen.items():
        model = "llama_3b" if label.startswith("llama") else "qwen_3b"
        base = frames[f"base_model/{model}_cot/gpqa_random_test"]
        df = frames[run].join(base[["response", "parsed_answer", "is_correct"]], rsuffix="_base")
        df.to_csv(OUT / f"{label}_paired_answers.csv")
        examples.append(f"## {run}\n")
        for row_id in example_ids[label]:
            row = df.loc[row_id]
            examples.append(f"### Test question {row_id}\n\n{row.question}\n\nOptions: {row.options}\n\n"
                f"Gold: {row.answer}; base: {row.parsed_answer_base}; new: {row.parsed_answer}.\n\n"
                f"New response:\n\n```text\n{row.response}\n```\n\n"
                f"Base response:\n\n```text\n{row.response_base}\n```\n")
    (OUT / "examples.md").write_text("\n".join(examples))

    import os
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gpqa-public-mpl-"))
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="gpqa-font-cache-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout="constrained")
    colors = ("#346ea2", "#ce8d2e", "#458452", "#944c89")
    for i, model in enumerate(("llama_3b", "qwen_3b")):
        base = metrics[metrics.model.eq(model) & metrics.arm.eq("base_model")].iloc[0]
        old = metrics[metrics.model.eq(model) & metrics.arm.str.startswith("corrected_reasoning")]
        for (arm, label), color in zip(ARMS.items(), colors):
            group = public[public.model.eq(model) & public.arm.eq(arm)].sort_values("epoch")
            if group.empty:
                continue
            axes[0, i].plot(group.epoch, 100 * group.accuracy, "o-", color=color, label=label)
            axes[1, i].plot(group.epoch, 100 * group.valid / 109, "o-", color=color)
            axes[2, i].plot(group.epoch, group.median_chars, "o-", color=color)
        axes[0, i].axhline(100 * base.accuracy, color="black", ls="--", lw=1, label="Original CoT")
        axes[0, i].axhline(100 * old.accuracy.max(), color="gray", ls=":", lw=1, label="Previous best")
        axes[1, i].axhline(100 * base.valid / 109, color="black", ls="--", lw=1)
        axes[2, i].axhline(base.median_chars, color="black", ls="--", lw=1)
        axes[0, i].set(title=model, ylim=(0, 45))
        axes[1, i].set(ylim=(30, 102))
        axes[2, i].set(xlabel="Epoch", ylim=(0, 10000))
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Valid final letters (%)")
    axes[2, 0].set_ylabel("Median output characters")
    axes[0, 1].legend(fontsize=8, loc="lower right")
    fig.suptitle("New GPQA public-mixture results — 109 held-out questions\nNo new Phi-4 Mini runs available")
    fig.savefig(OUT / "public_mix_overview.png", dpi=180)
    fig.savefig(OUT / "public_mix_overview.pdf")
    plt.close(fig)
    print(json.dumps({"checked_test_evaluations": len(metrics), "new_checkpoints": len(public),
        "missing_new_runs": missing, "best": [r for r in selections if r["selection"] == "best"]}, indent=2))


if __name__ == "__main__":
    main()
