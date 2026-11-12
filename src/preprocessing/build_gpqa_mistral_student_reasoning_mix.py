"""Extend the current GPQA public mix with Mistral and all three students.

Run from the repository root:
  .venv/bin/python src/preprocessing/build_gpqa_mistral_student_reasoning_mix.py

Inputs are read only. Correct final answers and source flags must agree; complete
targets must fit the character budget. No reasoning is truncated. Student traces
come from the base models' existing train_self_distill_*_cot exports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

if __package__:
    from .build_gpqa_public_reasoning_mix import (
        ROOT, SPLITS, Trace, build_mix, file_hash, normalize, option_list, option_text,
    )
else:
    from build_gpqa_public_reasoning_mix import (
        ROOT, SPLITS, Trace, build_mix, file_hash, normalize, option_list, option_text,
    )


OUTPUT = SPLITS / "train_reasoning_mix_public_mistral_students.parquet"
LOCAL_SOURCES = {
    "mistral_large_4_0": ("train_distilled_mistral_large_4_0.parquet", "mistralai/mistral-large-4-0"),
    "student_llama_3b": ("train_self_distill_llama_3b_cot.parquet", "llama_3b"),
    "student_phi4_mini": ("train_self_distill_phi4_mini_cot.parquet", "phi4_mini"),
    "student_qwen_3b": ("train_self_distill_qwen_3b_cot.parquet", "qwen_3b"),
}
ANSWER_ONLY = re.compile(r"\(?[a-d]\)?[.!]?", re.IGNORECASE)


def text_value(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def collect_local_traces(frame: pd.DataFrame, source: str, model: str) -> list[Trace]:
    if frame.empty:
        return []
    required = {"question", "options", "answer", "question_id", "distill_reasoning",
        "distill_answer", "distill_ans_correct"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    traces = []
    for row in frame.to_dict("records"):
        options = option_list(row["options"])
        reasoning = text_value(row["distill_reasoning"])
        answer = text_value(row["distill_answer"])
        exclusion = ""
        if text_value(row.get("corrected_reasoning")).strip():
            exclusion = "teacher_correction_excluded"
        elif ANSWER_ONLY.fullmatch(reasoning.strip()):
            exclusion = "answer_only_no_reasoning"
        traces.append(Trace(source, "base_model_cot" if source.startswith("student_") else "teacher_reasoning",
            str(row["question_id"]), text_value(row["question"]), reasoning,
            option_text(options, row["answer"]), options,
            source_correct=row["distill_ans_correct"] is True,
            source_answer=answer, written_answer=answer, split="our_train", model=model,
            verification="source_grade_and_exact_final_option_text", exclusion_reason=exclusion))
    return traces


def signature(row: dict) -> tuple:
    """Exact reasoning and final answer, with the question and option order fixed."""
    return (normalize(row["question"]), tuple(map(normalize, option_list(row["options"]))),
        row["distill_reasoning"].strip(), row["distill_answer"].strip().lower())


def extend_mix(train: pd.DataFrame, test: pd.DataFrame, previous: pd.DataFrame,
               additions: list[Trace], max_chars: int = 10000,
               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if previous.empty or not previous.question_id.is_unique:
        raise ValueError("Previous mixture must be nonempty with unique trace IDs")
    old_rows = previous.to_dict("records")
    old_traces = []
    for row in old_rows:
        if row["source"] == "current_corrected" and row["source_variant"] != "original_correct":
            raise ValueError("Previous mixture must exclude appended teacher corrections")
        options = option_list(row["options"])
        answer = text_value(row["distill_answer"])
        old_traces.append(Trace(row["source"], row["source_variant"], str(row["source_record_id"]),
            row["question"], row["distill_reasoning"], option_text(options, row["answer"]), options,
            source_correct=row["distill_ans_correct"] is True, source_answer=answer,
            written_answer=answer, split=row["source_split"], model=row["source_model"],
            verification=row["answer_verification"]))
    # The shared builder checks split membership by question text before grading,
    # aligns option text, enforces the complete-target budget and deduplicates.
    validated, audit = build_mix(train, test, [*old_traces, *additions], max_chars, seed)
    if not audit.iloc[:len(previous)].status.eq("included").all():
        failures = audit.iloc[:len(previous)].query("status != 'included'")
        raise ValueError(f"Previous mixture fails current checks: {failures.to_dict('records')}")
    # Keep every inherited row, including its ID and original source metadata,
    # exactly as it was. Only new rows receive new trace IDs.
    inherited = {signature(row): row for row in old_rows}
    result, added = [], 0
    for row in validated.to_dict("records"):
        if old := inherited.get(signature(row)):
            result.append(old)
        else:
            row["question_id"] = f"gpqa_mistral_students_mix_{added:05d}"
            result.append(row)
            added += 1
    out = pd.DataFrame(result)
    assert out.question_id.is_unique
    assert set(previous.question_id) <= set(out.question_id)
    pd.testing.assert_frame_equal(
        out.set_index("question_id").loc[previous.question_id, previous.columns.drop("question_id")],
        previous.set_index("question_id"), check_dtype=False)
    return out, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    paths = {"previous_mix": SPLITS / "train_reasoning_mix_public.parquet",
        "train": SPLITS / "train_original.parquet", "test": SPLITS / "test.parquet",
        **{source: SPLITS / name for source, (name, _) in LOCAL_SOURCES.items()}}
    if args.output.resolve() in {p.resolve() for p in paths.values()}:
        parser.error("Output must be separate from all inputs")
    hashes = {key: file_hash(path) for key, path in paths.items()}
    prior_manifest_path = paths["previous_mix"].with_suffix(".manifest.json")
    prior_manifest = json.loads(prior_manifest_path.read_text())
    if hashes["previous_mix"] != prior_manifest["output_sha256"]:
        raise ValueError("Previous mix does not match its manifest")
    for split in ("train", "test"):
        if hashes[split] != prior_manifest["input_files"][split]["sha256"]:
            raise ValueError(f"{split} split changed since the previous mixture was built")
    frames = {key: pd.read_parquet(path) for key, path in paths.items()}
    additions = []
    for source, (_, model) in LOCAL_SOURCES.items():
        additions.extend(collect_local_traces(frames[source], source, model))
    out, audit = extend_mix(frames["train"], frames["test"], frames["previous_mix"], additions,
        args.max_chars, args.seed)
    for key, path in paths.items():
        if file_hash(path) != hashes[key]:
            raise ValueError(f"Input changed during build: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    audit.to_csv(args.output.with_suffix(".audit.csv"), index=False)
    subset_dir = args.output.parent / f"{args.output.stem}_sources"
    subset_dir.mkdir(exist_ok=True)
    for source, rows in out.groupby("source"):
        rows.to_parquet(subset_dir / f"train_{source}.parquet", index=False)
    source_stats = out.groupby(["source", "source_variant"]).agg(
        rows=("question_id", "size"), questions=("original_question_id", "nunique"),
        median_response_chars=("response_chars", "median"),
        max_response_chars=("response_chars", "max")).reset_index().to_dict("records")
    display_path = lambda path: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    manifest = {"output": display_path(args.output), "rows": len(out),
        "previous_mix_rows": len(frames["previous_mix"]),
        "added_rows": len(out) - len(frames["previous_mix"]), "previous_mix_rows_preserved_exactly": True,
        "unique_training_questions": out.original_question_id.nunique(),
        "train_questions": len(frames["train"]), "test_questions": len(frames["test"]),
        "test_question_overlap": 0, "test_id_overlap": 0,
        "max_response_chars": int(out.response_chars.max()), "character_budget": args.max_chars,
        "budget_definition": "len('<think>' + distill_reasoning.strip() + '</think>' + distill_answer.strip())",
        "traces_truncated": 0, "seed": args.seed, "sources": source_stats,
        "candidate_outcomes": audit.groupby(["source", "status"]).size().reset_index(name="rows").to_dict("records"),
        "input_files": {key: {"path": str(path.relative_to(ROOT)), "sha256": hashes[key]}
            for key, path in paths.items()},
        "previous_mix_manifest": prior_manifest, "output_sha256": file_hash(args.output),
        "source_subsets": display_path(subset_dir),
        "notes": ["Every previous mixture row and ID is retained unchanged; inputs are read only.",
            "All additions require both a true source correctness flag and a final option matching the local training gold.",
            "Student traces are the existing correct base-model CoT exports, shared across all three students.",
            "Answer-only placeholders and appended teacher corrections are excluded.",
            "Exact duplicates require the same question, option order, reasoning and final letter; distinct chains are retained.",
            "Correct final labels do not certify every intermediate reasoning step."]}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("output", "rows", "previous_mix_rows", "added_rows",
        "unique_training_questions", "test_question_overlap", "max_response_chars", "sources", "candidate_outcomes")}, indent=2))


if __name__ == "__main__":
    main()
