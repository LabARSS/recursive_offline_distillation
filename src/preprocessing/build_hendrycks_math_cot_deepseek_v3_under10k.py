"""Build the DeepSeek V3 CoT MATH pool keyed by train UUID question IDs.

One chain per train question (verified correct against train gold, strictly under
10,000 characters, never truncated, test-overlap removed, trailing ``Answer: <x>``
line dropped). question_id is the train UUID, so proxy entropy estimated on
hendrycks_math_train merges one-to-one.

Run from the repository root (after preprocessing.assign_hendrycks_math_train_uuids):
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_cot_deepseek_v3_under10k
"""

import json

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import DISTILL, ROOT, build_mix, sha256_file
from .build_hendrycks_math_reasoning_mix_10k_cot_teachers import length_stats, teacher_traces

TEACHER = "deepseek_v3"
OUTPUT = DISTILL / f"hendrycks_math_distilled_cot_{TEACHER}_under10k.parquet"


def main() -> None:
    paths = {
        "train": ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet",
        "test": ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet",
        TEACHER: DISTILL / f"hendrycks_math_distilled_cot_{TEACHER}.parquet",
    }
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    frame, audit, _ = build_mix(
        frames["train"], frames["test"], teacher_traces(TEACHER, frames[TEACHER]), max_chars=10000
    )
    # Whitespace-equivalent train questions share the canonical ID; keep the canonical row's own chain.
    duplicated = frame.question_id.duplicated(keep=False)
    frame = frame[~duplicated | frame.source_record_id.eq(frame.question_id)].reset_index(drop=True)
    assert frame.question_id.is_unique and frame.question_id.isin(frames["train"].question_id).all()
    assert not frame.distill_reasoning.str.contains(r"\nAnswer:[^\n]*$", regex=True).any()

    frame.to_parquet(OUTPUT, index=False)
    audit.to_parquet(OUTPUT.with_suffix(".audit.parquet"), index=False)
    manifest = {
        "rows": len(frame), "source": TEACHER, **length_stats(frame.reasoning_chars),
        "filter_counts": audit.status.value_counts().to_dict(),
        "question_ids": "train UUIDs, one chain per question",
        "input_files": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "output_sha256": sha256_file(OUTPUT),
    }
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "median_chars", "p95_chars", "filter_counts")}, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
