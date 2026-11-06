"""Build a MATH mix dominated by same-family CoT teachers (Llama 70B, Qwen 72B).

Sources (all chains verified correct, strictly under 10,000 characters, never
truncated, test-overlap removed):
  * Llama-3.3-70B and Qwen2.5-72B CoT: all eligible chains.
  * DeepSeek V3 CoT: a random 50% of eligible chains.
  * OpenMathInstruct-2 original-MATH chains: a random 25% of eligible chains.

Teacher chains are re-graded against train gold (the stored Llama-70B flags
were format-broken before ``fix_llama_70b_cot_answer_format``). Their trailing
``Answer: <x>`` line is dropped from the reasoning, since the trainer appends
the answer after </think> itself. Every row gets a unique question_id;
original_question_id keeps the train question grouping.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_reasoning_mix_10k_cot_teachers
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .build_hendrycks_math_reasoning_mix_10k import (
    DISTILL, ROOT, Trace, build_mix, prediction_from_response, sha256_file,
)

OUTPUT = DISTILL / "hendrycks_math_reasoning_mix_under10k_cot_teachers.parquet"
TEACHERS = ("llama_70b", "qwen_72b", "deepseek_v3")
SAMPLE_FRACTIONS = {"omi2_math": 0.25, "deepseek_v3": 0.5}
SEED = 42
ANSWER_LINE = re.compile(r"(?:\n[ \t]*Answer:[^\n]*)+\s*$")


def omi2_traces(omi2: pd.DataFrame) -> Iterable[Trace]:
    for idx, row in enumerate(omi2.to_dict("records")):
        if row["problem_source"] != "math":
            continue
        reasoning = row["distill_reasoning"]
        yield Trace(
            "omi2_math", f"omi2_20k:{row.get('question_id', idx)}", row["question"], reasoning,
            prediction_from_response(reasoning), "distill_reasoning", row["distill_ans_correct"] is True,
        )


def teacher_traces(name: str, frame: pd.DataFrame) -> Iterable[Trace]:
    for row in frame.to_dict("records"):
        response = row["distill_answer"] if isinstance(row["distill_answer"], str) else ""
        yield Trace(
            name, str(row["question_id"]), row["question"], ANSWER_LINE.sub("", response.rstrip()),
            prediction_from_response(response), "distill_answer[:-answer_line]",
        )


def length_stats(lengths: pd.Series) -> dict:
    return {
        "mean_chars": float(lengths.mean()), "median_chars": float(lengths.median()),
        **{f"p{q}_chars": float(lengths.quantile(q / 100)) for q in (90, 95, 99)},
        "max_chars": int(lengths.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    paths = {
        "train": ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet",
        "test": ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet",
        "omi2_20k": DISTILL / "hendrycks_math_openmathinstruct2_math_all2sol_aug7000.parquet",
        **{name: DISTILL / f"hendrycks_math_distilled_cot_{name}.parquet" for name in TEACHERS},
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}

    parts, audits, eligible = [], [], {}
    sources = {"omi2_math": omi2_traces(frames["omi2_20k"])}
    sources |= {name: teacher_traces(name, frames[name]) for name in TEACHERS}
    for name, traces in sources.items():
        frame, audit, _ = build_mix(frames["train"], frames["test"], traces, max_chars=10000)
        eligible[name] = len(frame)
        if name in SAMPLE_FRACTIONS:
            kept = frame.sort_values("trace_id").sample(frac=SAMPLE_FRACTIONS[name], random_state=SEED)
            audit.loc[audit.status.eq("included") & ~audit.reasoning_sha256.isin(kept.reasoning_sha256), "status"] = "not_sampled"
            frame = kept
        print(f"{name}: {audit.status.value_counts().to_dict()}", flush=True)
        parts.append(frame)
        audits.append(audit)
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"Input changed during construction: {path}")

    output = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    output = output.drop_duplicates(["question_id", "reasoning_sha256"]).reset_index(drop=True)
    output["original_question_id"] = output.question_id
    output["question_id"] = [f"math_cot_teachers_{i:06d}" for i in range(len(output))]
    assert output.trace_id.is_unique and output.distill_reasoning.str.len().lt(10000).all()
    teachers = output[output.source.isin(TEACHERS)]
    assert not teachers.distill_reasoning.str.contains(r"\nAnswer:[^\n]*$", regex=True).any()

    manifest = {
        "rows": len(output), "unique_questions": output.original_question_id.nunique(),
        **length_stats(output.reasoning_chars),
        "sample_fractions": SAMPLE_FRACTIONS, "seed": SEED, "eligible_before_sampling": eligible,
        "max_reasoning_chars_exclusive": 10000, "traces_truncated": 0,
        "correctness": "OMI2: source flag plus final-answer verification; CoT teachers: final-answer verification against train gold",
        "teacher_reasoning": "CoT response with its trailing 'Answer: <x>' line removed; the answer follows </think> in training",
        "decontamination": "exact match or shared 13-word sequence with all 5,000 MATH test questions",
        "question_ids": "unique per row; original_question_id retains question grouping only",
        "sources": {
            source: {"traces": len(rows), "unique_questions": rows.original_question_id.nunique(), **length_stats(rows.reasoning_chars)}
            for source, rows in output.groupby("source")
        },
        "chains_per_question": {str(k): int(v) for k, v in output.groupby("original_question_id").size().value_counts().sort_index().items()},
        "input_files": {name: {"path": str(path.relative_to(ROOT)), "sha256": hashes[name]} for name, path in paths.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    pd.concat(audits, ignore_index=True).to_parquet(args.output.with_suffix(".audit.parquet"), index=False)
    manifest["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "unique_questions", "median_chars", "p95_chars", "sources", "chains_per_question")}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
