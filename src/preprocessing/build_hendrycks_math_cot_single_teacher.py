"""Split the CoT-teacher MATH mix into single-teacher datasets (Qwen 72B, Llama 70B).

Rows are taken verbatim from hendrycks_math_reasoning_mix_under10k_cot_teachers,
which already holds every eligible chain of both teachers (verified correct,
under 10,000 characters, test-decontaminated, trailing Answer line removed),
so the single-teacher runs differ from the mix runs only in which sources remain.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_cot_single_teacher
"""

import json

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import DISTILL, sha256_file
from .build_hendrycks_math_reasoning_mix_10k_cot_teachers import OUTPUT as MIX

TEACHERS = ("qwen_72b", "llama_70b")


def main() -> None:
    mix = pd.read_parquet(MIX)
    for teacher in TEACHERS:
        frame = mix[mix.source == teacher].reset_index(drop=True)
        assert len(frame) and frame.trace_id.is_unique
        out = DISTILL / f"hendrycks_math_cot_{teacher}_under10k.parquet"
        frame.to_parquet(out, index=False)
        lengths = frame.reasoning_chars
        manifest = {
            "rows": len(frame), "unique_questions": frame.original_question_id.nunique(), "source": teacher,
            "mean_chars": float(lengths.mean()), "median_chars": float(lengths.median()),
            **{f"p{q}_chars": float(lengths.quantile(q / 100)) for q in (90, 95, 99)},
            "derived_from": {"path": MIX.name, "sha256": sha256_file(MIX)}, "output_sha256": sha256_file(out),
        }
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{teacher}: {len(frame):,} rows -> {out}")


if __name__ == "__main__":
    main()
