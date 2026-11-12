"""Drop the student self chains from the Qwen-style and Llama-style MATH mixes.

Rows are taken verbatim from the style mixes (verified correct, under 10,000
characters, test-decontaminated, cleaned answers), so these runs differ from
the style-mix runs only in which sources remain:
  * qwen_72b_mistral: Qwen2.5-72B CoT + filtered Mistral Medium CoT.
  * llama_70b_omi2: Llama-3.3-70B CoT + OpenMathInstruct-2 (one chain per question).

Run from the repository root (after the two style-mix builders):
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_reasoning_mix_10k_style_teachers
"""

import json

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import DISTILL, sha256_file
from .build_hendrycks_math_reasoning_mix_10k_cot_teachers import length_stats

MIXES = {
    "qwen_72b_mistral": ("hendrycks_math_reasoning_mix_under10k_qwen_style", {"qwen_72b", "mistral_medium"}),
    "llama_70b_omi2": ("hendrycks_math_reasoning_mix_under10k_llama_style", {"llama_70b", "omi2_math"}),
}


def main() -> None:
    for name, (parent, sources) in MIXES.items():
        source_path = DISTILL / f"{parent}.parquet"
        mix = pd.read_parquet(source_path)
        frame = mix[mix.source.isin(sources)].reset_index(drop=True)
        assert set(frame.source) == sources and frame.trace_id.is_unique and frame.question_id.is_unique
        out = DISTILL / f"hendrycks_math_reasoning_mix_under10k_{name}.parquet"
        frame.to_parquet(out, index=False)
        manifest = {
            "rows": len(frame), "unique_questions": frame.original_question_id.nunique(),
            **length_stats(frame.reasoning_chars),
            "sources": {
                source: {"traces": len(rows), "unique_questions": rows.original_question_id.nunique(),
                         **length_stats(rows.reasoning_chars)}
                for source, rows in frame.groupby("source")
            },
            "chains_per_question": {
                str(k): int(v)
                for k, v in frame.groupby("original_question_id").size().value_counts().sort_index().items()
            },
            "derived_from": {"path": source_path.name, "sha256": sha256_file(source_path), "kept_sources": sorted(sources)},
            "output_sha256": sha256_file(out),
        }
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(frame):,} rows over {manifest['unique_questions']:,} questions; "
              f"{ {s: v['traces'] for s, v in manifest['sources'].items()} }; "
              f"median {manifest['median_chars']:.0f}, p95 {manifest['p95_chars']:.0f}; "
              f"chains/question {manifest['chains_per_question']} -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
