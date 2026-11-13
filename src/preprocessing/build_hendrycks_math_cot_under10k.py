"""Build CoT MATH pools from local teachers, keyed by train UUID question IDs.

At most one chain per (teacher, train question): strictly under 10,000 characters,
never truncated, non-empty, with an extractable final answer, test-overlap removed,
trailing ``Answer: <x>`` line dropped. By default chains must be verified correct
against train gold; --include-incorrect keeps wrong-answer chains as well
(distill_ans_correct records the grade). question_id is the train UUID, so proxy
entropy estimated on hendrycks_math_train merges onto single-teacher pools; with
several teachers a question_id repeats once per teacher.

Run from the repository root (after preprocessing.assign_hendrycks_math_train_uuids):
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_cot_under10k
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_cot_under10k --teachers deepseek_v3 llama_70b --include-incorrect
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import DISTILL, ROOT, build_mix, sha256_file
from .build_hendrycks_math_reasoning_mix_10k_cot_teachers import length_stats, teacher_traces

TEACHERS = ("deepseek_v3", "llama_70b", "qwen_72b", "mistral_medium")


def output_path(teachers: list[str], include_incorrect: bool) -> Path:
    suffix = "_w_incorrect" if include_incorrect else ""
    return DISTILL / f"hendrycks_math_distilled_cot_{'_'.join(teachers)}_under10k{suffix}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", nargs="+", choices=TEACHERS, default=["deepseek_v3"])
    parser.add_argument("--include-incorrect", action="store_true")
    args = parser.parse_args()
    output = output_path(args.teachers, args.include_incorrect)

    paths = {
        "train": ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet",
        "test": ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet",
        **{teacher: DISTILL / f"hendrycks_math_distilled_cot_{teacher}.parquet" for teacher in args.teachers},
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}

    parts, audits = [], []
    for teacher in args.teachers:
        frame, audit, _ = build_mix(
            frames["train"], frames["test"], teacher_traces(teacher, frames[teacher]),
            max_chars=10000, require_correct=not args.include_incorrect,
        )
        # Whitespace-equivalent train questions share the canonical ID; keep the canonical row's own chain.
        duplicated = frame.question_id.duplicated(keep=False)
        frame = frame[~duplicated | frame.source_record_id.eq(frame.question_id)]
        assert frame.question_id.is_unique and frame.question_id.isin(frames["train"].question_id).all()
        print(f"{teacher}: {audit.status.value_counts().to_dict()}", flush=True)
        parts.append(frame)
        audits.append(audit)

    frame = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    assert frame.trace_id.is_unique
    assert not frame.distill_reasoning.str.strip().eq("").any()
    assert not frame.distill_reasoning.str.contains(r"\nAnswer:[^\n]*$", regex=True).any()

    frame.to_parquet(output, index=False)
    audit = pd.concat(audits, ignore_index=True)
    audit.to_parquet(output.with_suffix(".audit.parquet"), index=False)
    manifest = {
        "rows": len(frame), "unique_questions": frame.question_id.nunique(),
        "correct_fraction": float(frame.distill_ans_correct.astype(bool).mean()),
        "include_incorrect": args.include_incorrect, **length_stats(frame.reasoning_chars),
        "sources": {
            source: {"traces": len(rows), "correct": int(rows.distill_ans_correct.astype(bool).sum()),
                     "filter_counts": audit[audit.source == source].status.value_counts().to_dict()}
            for source, rows in frame.groupby("source")
        },
        "question_ids": "train UUIDs, one chain per question per teacher",
        "input_files": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "unique_questions", "correct_fraction", "median_chars", "p95_chars")}, indent=2))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
