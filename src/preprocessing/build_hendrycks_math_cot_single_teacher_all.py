"""Build single-teacher MATH CoT pools (Qwen 72B, Llama 70B) without a correctness filter.

Unlike hendrycks_math_cot_<teacher>_under10k, wrong-answer chains are kept: the
student trains on the teacher's own final answer, right or wrong
(distill_ans_correct records which). Filters kept: under 10,000 characters,
test-decontaminated, non-empty, no control markers, and an extractable final
answer (the answer follows </think> in training). The trailing ``Answer: <x>``
line is removed from the reasoning, as in the filtered pools.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_cot_single_teacher_all
"""

import json

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import DISTILL, ROOT, build_mix, sha256_file
from .build_hendrycks_math_reasoning_mix_10k_cot_teachers import length_stats, teacher_traces

TEACHERS = ("qwen_72b", "llama_70b")
MAX_CHARS = 10000


def main() -> None:
    paths = {
        "train": ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet",
        "test": ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet",
        **{t: DISTILL / f"hendrycks_math_distilled_cot_{t}.parquet" for t in TEACHERS},
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    train, test = pd.read_parquet(paths["train"]), pd.read_parquet(paths["test"])

    for teacher in TEACHERS:
        traces = teacher_traces(teacher, pd.read_parquet(paths[teacher]))
        frame, audit, _ = build_mix(train, test, traces, max_chars=MAX_CHARS, require_correct=False)
        frame["original_question_id"] = frame.question_id
        frame["question_id"] = [f"math_cot_{teacher}_all_{i:06d}" for i in range(len(frame))]
        assert frame.trace_id.is_unique and frame.distill_reasoning.str.len().lt(MAX_CHARS).all()
        assert not frame.distill_reasoning.str.contains(r"\nAnswer:[^\n]*$", regex=True).any()
        assert frame.distill_answer.str.strip().ne("").all()

        out = DISTILL / f"hendrycks_math_cot_{teacher}_all_under10k.parquet"
        frame.to_parquet(out, index=False)
        audit.to_parquet(out.with_suffix(".audit.parquet"), index=False)
        manifest = {
            "rows": len(frame), "unique_questions": frame.original_question_id.nunique(), "source": teacher,
            "correct": int(frame.distill_ans_correct.sum()), "incorrect": int((~frame.distill_ans_correct).sum()),
            **length_stats(frame.reasoning_chars),
            "audit_status": {k: int(v) for k, v in audit.status.value_counts().items()},
            "max_reasoning_chars_exclusive": MAX_CHARS, "traces_truncated": 0,
            "correctness": "not filtered; distill_ans_correct = final answer verified against train gold",
            "decontamination": "exact match or shared 13-word sequence with all 5,000 MATH test questions",
            "question_ids": "unique per row; original_question_id retains question grouping only",
            "input_files": {
                name: {"path": str(paths[name].relative_to(ROOT)), "sha256": hashes[name]}
                for name in ("train", "test", teacher)
            },
            "output_sha256": sha256_file(out),
        }
        out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{teacher}: {len(frame):,} rows ({manifest['incorrect']:,} incorrect) -> {out}", flush=True)

    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"Input changed during construction: {path}")


if __name__ == "__main__":
    main()
