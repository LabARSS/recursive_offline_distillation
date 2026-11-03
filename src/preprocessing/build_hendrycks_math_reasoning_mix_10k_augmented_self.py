"""Extend the under-10k MATH mix with OMI2 20k and student self-distillation.

Retain every existing chain, add eligible OMI2 chains (including augmented
problems), and add all stored Llama, Phi, and Qwen self-distillation chains.
Require verified final answers and reasoning strictly below 10,000 characters;
never truncate. Duplicate question/chain pairs and test overlaps are excluded.

Every output row receives a fresh, unique question_id. original_question_id
retains the question grouping as provenance; augmented questions use an OMI2
namespace. The existing mixture and all input files are read only.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_reasoning_mix_10k_augmented_self
"""

import argparse
from itertools import chain
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import (
    DISTILL, OUTPUT as PREVIOUS_MIX, ROOT, Trace, build_mix, correct_answer,
    prediction_from_response, question_key, sha256_file,
)

OUTPUT = DISTILL / "hendrycks_math_reasoning_mix_under10k_omi2_aug_self.parquet"
SELF_MODELS = ("llama_3b", "phi4_mini", "qwen_3b")


def training_catalog(train: pd.DataFrame, omi2: pd.DataFrame) -> pd.DataFrame:
    """Add references for augmented questions absent from the original train."""
    references = {question_key(row["question"]): row for row in train.to_dict("records")}
    additions = []
    for idx, row in enumerate(omi2.to_dict("records")):
        if row["problem_source"] != "augmented_math":
            continue
        key = question_key(row["question"])
        if key in references:
            if not correct_answer(str(references[key]["answer"]), str(row["answer"])):
                raise ValueError("An augmented question has inconsistent reference answers")
            continue
        reference = {
            "question_id": f"omi2_augmented_{row.get('question_id', idx)}",
            "question": row["question"], "answer": str(row["answer"]),
            "answer_w_steps": row.get("answer_w_steps", ""),
        }
        references[key] = reference
        additions.append(reference)
    return pd.concat([train, pd.DataFrame(additions)], ignore_index=True)


def previous_traces(previous: pd.DataFrame) -> Iterable[Trace]:
    for row in previous.to_dict("records"):
        yield Trace(
            row["source"], row["source_record_id"], row["question"],
            row["distill_reasoning"], row["distill_answer"],
            row["source_reasoning_column"], row["distill_ans_correct"] is True,
        )


def omi2_additions(omi2: pd.DataFrame) -> Iterable[Trace]:
    for idx, row in enumerate(omi2.to_dict("records")):
        subset = row["problem_source"]
        if subset not in {"math", "augmented_math"}:
            continue
        reasoning = row["distill_reasoning"]
        yield Trace(
            "omi2_augmented_math" if subset == "augmented_math" else "omi2_math",
            f"omi2_20k:{row.get('question_id', idx)}", row["question"], reasoning,
            prediction_from_response(reasoning), "distill_reasoning",
            row["distill_ans_correct"] is True,
        )


def self_additions(students: dict[str, pd.DataFrame]) -> Iterable[Trace]:
    for model, frame in students.items():
        for row in frame.to_dict("records"):
            yield Trace(
                f"self_{model}", f"{row['question_id']}:{row.get('pass', 'unknown')}",
                row["question"], row["distill_reasoning"], row["distill_answer"],
                "distill_reasoning", row["distill_ans_correct"] is True,
            )


def extend_mix(train: pd.DataFrame, test: pd.DataFrame, previous: pd.DataFrame,
               omi2: pd.DataFrame, students: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if previous.empty or not previous.trace_id.is_unique:
        raise ValueError("The existing mixture must have unique trace IDs and contain at least one row")
    catalog = training_catalog(train, omi2)
    output, audit, _ = build_mix(
        catalog, test,
        chain(previous_traces(previous), omi2_additions(omi2), self_additions(students)),
        max_chars=10000, progress=True,
    )
    if not audit.iloc[:len(previous)].status.eq("included").all():
        failures = audit.iloc[:len(previous)].query("status != 'included'")
        raise ValueError(f"An inherited chain failed validation: {failures.to_dict('records')}")

    # The flat OMI2 export strips surrounding whitespace, while the inherited
    # raw-shard traces preserve it. These are the same chain for training.
    inherited_ids = set(previous.trace_id)
    seen = {(question_key(row.question), row.distill_reasoning.strip()) for row in previous.itertuples()}
    removed_ids, duplicate_records = set(), set()
    for row in output.itertuples():
        if row.trace_id in inherited_ids:
            continue
        signature = (question_key(row.question), row.distill_reasoning.strip())
        if signature in seen:
            removed_ids.add(row.trace_id)
            duplicate_records.add((row.source, row.source_record_id, row.reasoning_sha256))
        else:
            seen.add(signature)
    output = output[~output.trace_id.isin(removed_ids)].copy()
    duplicate_mask = [
        row.status == "included" and (row.source, row.source_record_id, row.reasoning_sha256) in duplicate_records
        for row in audit.itertuples()
    ]
    audit.loc[duplicate_mask, "status"] = "duplicate_chain"

    # Each inherited payload and its provenance must survive exactly; only the
    # question_id column is intentionally replaced by a unique per-row ID.
    previous_columns = previous.columns.drop("question_id")
    pd.testing.assert_frame_equal(
        output.set_index("trace_id").loc[previous.trace_id, previous_columns.drop("trace_id")],
        previous.set_index("trace_id")[previous_columns.drop("trace_id")], check_dtype=False,
    )
    output["original_question_id"] = output.question_id
    output["question_id"] = [f"math_aug_self_{i:06d}" for i in range(len(output))]
    audit = audit.rename(columns={"question_id": "original_question_id"})
    assert output.question_id.is_unique and output.trace_id.is_unique
    assert output.distill_reasoning.str.len().lt(10000).all()
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = {
        "previous_mix": PREVIOUS_MIX,
        "train": ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet",
        "test": ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet",
        "omi2_20k": DISTILL / "hendrycks_math_openmathinstruct2_math_all2sol_aug7000.parquet",
        **{f"self_{model}": DISTILL / f"hendrycks_math_self_distilled_{model}.parquet" for model in SELF_MODELS},
    }
    args.output = args.output.resolve()
    if args.output in {path.resolve() for path in paths.values()}:
        parser.error("Output must be separate from every input")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    students = {model: frames[f"self_{model}"] for model in SELF_MODELS}
    output, audit = extend_mix(
        frames["train"], frames["test"], frames["previous_mix"], frames["omi2_20k"], students,
    )
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"Input changed during construction: {path}")

    inherited_ids = set(frames["previous_mix"].trace_id)
    added = output[~output.trace_id.isin(inherited_ids)]
    sources = {}
    for source, rows in output.groupby("source"):
        sources[source] = {
            "traces": len(rows), "unique_questions": rows.original_question_id.nunique(),
            "mean_chars": float(rows.reasoning_chars.mean()),
            "median_chars": float(rows.reasoning_chars.median()),
        }
    manifest = {
        "rows": len(output), "unique_question_ids": output.question_id.nunique(),
        "unique_questions": output.original_question_id.nunique(),
        "inherited_rows": len(frames["previous_mix"]), "added_rows": len(added),
        "added_by_source": added.source.value_counts().to_dict(),
        "mean_reasoning_chars": float(output.reasoning_chars.mean()),
        "median_reasoning_chars": float(output.reasoning_chars.median()),
        "max_reasoning_chars": int(output.reasoning_chars.max()),
        "max_reasoning_chars_exclusive": 10000, "traces_truncated": 0,
        "question_ids": "unique per row; original_question_id retains question grouping only",
        "inherited_payloads_and_provenance_preserved": True,
        "correctness": "source correctness flags plus independent final-answer verification",
        "omi2_reference_answers": "original train gold for original MATH questions; OMI2 reference answers for augmented questions",
        "decontamination": "exact match or shared 13-word sequence with all 5,000 MATH test questions",
        "deduplication": "additions must have a new question/chain pair; surrounding chain whitespace is ignored for comparison only",
        "self_distillation_passes": {model: frame["pass"].value_counts().to_dict() for model, frame in students.items()},
        "sources": sources,
        "candidate_outcomes": audit.groupby(["source", "status"]).size().reset_index(name="rows").to_dict("records"),
        "input_files": {name: {"path": str(path.relative_to(ROOT)), "sha256": hashes[name]} for name, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    audit.to_parquet(args.output.with_suffix(".audit.parquet"), index=False)
    manifest["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in [
        "rows", "unique_question_ids", "unique_questions", "inherited_rows", "added_rows",
        "added_by_source", "mean_reasoning_chars", "max_reasoning_chars", "sources",
    ]}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
