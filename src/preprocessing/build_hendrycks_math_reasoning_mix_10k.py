"""Build five-source MATH reasoning data with original train question IDs.

Keep verified-answer chains strictly shorter than 10,000 characters, without
truncation. Sources: all original-MATH rows in OpenMathInstruct-2 train_1M,
Qwen3 math8k, sxiong/MATH-trajectory, local regenerated DeepSeek traces, and
the written distill_answer from local extra-detailed regenerated DeepSeek data.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_reasoning_mix_10k

Public inputs are read from existing caches, or explicit --omi2 / --qwen3 /
--sxiong paths. No network access is performed. --sxiong accepts parquet or
JSONL. An audit parquet and JSON manifest accompany the training parquet.
"""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from huggingface_hub import try_to_load_from_cache

from core.utils.cot_answers import extract_math_answer, verify_math_answer

from .download_hendrycks_math import extract_final_answer
from .download_openmathinstruct2 import ngrams, normalize as decontamination_key

ROOT = Path(__file__).resolve().parents[2]
DISTILL = ROOT / "data/out/distillation"
OUTPUT = DISTILL / "hendrycks_math_reasoning_mix_omi2_qwen3_sxiong_deepseek_under10k.parquet"
SXIONG_PROBE = ROOT / (
    "artifacts/distillation_on_synthetic_traces/hendrycks_math/analysis/"
    "public_math_traces/sxiong__MATH-trajectory__default__0.parquet"
)
OUTPUT_COLUMNS = [
    "question_id", "question", "answer", "answer_w_steps", "distill_reasoning",
    "distill_answer", "distill_ans_correct", "source", "source_record_id",
    "source_reasoning_column", "trace_id", "reasoning_chars", "reasoning_sha256",
]


def question_key(text: str) -> str:
    """Ignore whitespace differences, retaining mathematical symbol case."""
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def correct_answer(gold: str, prediction: str) -> bool:
    return verify_math_answer(gold, prediction)


def prediction_from_response(response: str) -> str:
    if not isinstance(response, str):
        return ""
    prediction = extract_math_answer(response, ())
    if prediction:
        return prediction
    try:
        return extract_final_answer(response)
    except ValueError:
        return ""


@dataclass(frozen=True)
class Trace:
    source: str
    source_record_id: str
    question: str
    reasoning: str
    prediction: str
    reasoning_column: str
    source_correct: bool | None = None


def omi2_traces(frame: pd.DataFrame, record_prefix: str = "") -> Iterable[Trace]:
    """Grade the generated solution itself, not its expected-answer label."""
    raw = "problem" in frame.columns
    for idx, row in frame.iterrows():
        if row["problem_source"] != "math":
            continue
        reasoning = row["generated_solution" if raw else "distill_reasoning"]
        yield Trace(
            "omi2_math", f"{record_prefix}{idx}", row["problem" if raw else "question"],
            reasoning, prediction_from_response(reasoning),
            "generated_solution" if raw else "distill_reasoning",
        )


def qwen3_traces(frame: pd.DataFrame) -> Iterable[Trace]:
    for idx, row in frame.iterrows():
        replies = [message["content"] for message in row["conversations"] if message["role"] == "assistant"]
        reply = replies[-1] if replies else ""
        thinking, separator, final = reply.partition("</think>")
        # Remove the container marker, preserving the complete reasoning payload.
        reasoning = thinking.removeprefix("<think>") if separator else ""
        yield Trace(
            "qwen3_math8k", str(idx), row["problem"], reasoning,
            prediction_from_response(final) if separator else "", "conversations.assistant.think",
        )


def sxiong_traces(frame: pd.DataFrame) -> Iterable[Trace]:
    for _, row in frame.iterrows():
        yield Trace(
            "sxiong_r1", str(row["id"]), row["question"], row["reasoning_content"],
            prediction_from_response(row["content"]), "reasoning_content", bool(row["correct"]),
        )


def deepseek_traces(frame: pd.DataFrame, *, extra_detailed: bool = False) -> Iterable[Trace]:
    column = "distill_answer" if extra_detailed else "distill_reasoning"
    source = "deepseek_extra_detailed_written" if extra_detailed else "deepseek_regenerated"
    for idx, row in frame.iterrows():
        prediction = prediction_from_response(row["distill_answer"]) if extra_detailed else row["distill_answer"]
        yield Trace(
            source, str(idx), row["question"], row[column], prediction, column,
            bool(row["distill_ans_correct"]),
        )


def train_index(train: pd.DataFrame) -> tuple[dict[str, dict], list[dict]]:
    if train.question_id.astype(str).duplicated().any():
        raise ValueError("Original train question IDs must be unique")
    index, aliases = {}, []
    for row in train.to_dict("records"):
        row["question_id"] = str(row["question_id"])
        key = question_key(row["question"])
        if key in index:
            canonical = index[key]
            if not correct_answer(str(canonical["answer"]), str(row["answer"])):
                raise ValueError("Equivalent train questions have inconsistent gold answers")
            aliases.append({"question_id": row["question_id"], "canonical_question_id": canonical["question_id"]})
        else:
            index[key] = row
    return index, aliases


def build_mix(train: pd.DataFrame, test: pd.DataFrame, traces: Iterable[Trace], *, max_chars: int = 10000,
              progress: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    index, aliases = train_index(train)
    test_questions = set(test.question.map(decontamination_key))
    test_ngrams = set().union(*(ngrams(question) for question in test_questions))

    @lru_cache(maxsize=None)
    def contaminated(question: str) -> bool:
        key = decontamination_key(question)
        return key in test_questions or not ngrams(key).isdisjoint(test_ngrams)

    output, audit, seen = [], [], set()
    for ordinal, trace in enumerate(traces, 1):
        if progress and ordinal % 1000 == 0:
            print(f"Processed {ordinal:,} candidate traces; included {len(output):,}", flush=True)
        reasoning = trace.reasoning if isinstance(trace.reasoning, str) else ""
        prediction = trace.prediction.strip() if isinstance(trace.prediction, str) else ""
        digest = hashlib.sha256(reasoning.encode()).hexdigest()
        canonical = index.get(question_key(trace.question))
        qid = canonical["question_id"] if canonical else None
        record = {
            "source": trace.source, "source_record_id": trace.source_record_id,
            "source_reasoning_column": trace.reasoning_column, "question_id": qid,
            "reasoning_chars": len(reasoning), "reasoning_sha256": digest,
        }
        status = "included"
        if contaminated(trace.question):
            status = "test_overlap"
        elif canonical is None:
            status = "not_in_original_train"
        elif not reasoning.strip():
            status = "empty_reasoning"
        elif len(reasoning) >= max_chars:
            status = "at_or_over_character_limit"
        elif trace.source_correct is False:
            status = "source_marked_incorrect"
        elif any(marker in reasoning for marker in ("<think>", "</think>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>")):
            status = "embedded_control_marker"
        elif not prediction:
            status = "missing_final_answer"
        elif not correct_answer(str(canonical["answer"]), prediction):
            status = "final_answer_incorrect"
        elif (qid, digest) in seen:
            status = "duplicate_chain"
        if status == "included":
            seen.add((qid, digest))
            trace_id = hashlib.sha256(f"{qid}\0{reasoning}\0{prediction}".encode()).hexdigest()[:24]
            output.append({
                **record, "question": canonical["question"], "answer": str(canonical["answer"]),
                "answer_w_steps": canonical.get("answer_w_steps", ""),
                "distill_reasoning": reasoning, "distill_answer": prediction,
                "distill_ans_correct": True, "trace_id": trace_id,
            })
        audit.append({**record, "status": status})
    frame = pd.DataFrame(output, columns=OUTPUT_COLUMNS)
    if not frame.empty:
        frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
        assert frame.trace_id.is_unique
        assert frame.distill_reasoning.str.len().lt(max_chars).all()
        assert frame.groupby("question_id").question.nunique().eq(1).all()
        assert frame.groupby("question").question_id.nunique().eq(1).all()
        gold = train.set_index(train.question_id.astype(str))
        assert all(gold.loc[row.question_id, "question"] == row.question for row in frame.itertuples())
    return frame, pd.DataFrame(audit), aliases


def cached_file(repo: str, filename: str) -> Path:
    path = try_to_load_from_cache(repo, filename, repo_type="dataset")
    if not isinstance(path, str):
        raise FileNotFoundError(f"No cached {repo}/{filename}; provide the corresponding input path")
    return Path(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omi2", type=Path, nargs="+", help="Raw OMI2 shards or a flat OMI2 parquet")
    parser.add_argument("--qwen3", type=Path)
    parser.add_argument("--sxiong", type=Path, default=SXIONG_PROBE)
    parser.add_argument("--deepseek", type=Path, default=DISTILL / "hendrycks_math_distilled_deepseek_v4_flash_regenerate_incorrect_w_large.parquet")
    parser.add_argument("--extra-detailed", type=Path, default=DISTILL / "hendrycks_math_distilled_extra_detailed_raw_deepseek_v4_flash_regenerate_incorrect_w_large.parquet")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    train_path = ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet"
    test_path = ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet"
    omi_paths = args.omi2 or [cached_file("nvidia/OpenMathInstruct-2", f"data/train_1M-{i:05d}-of-00003.parquet") for i in range(3)]
    qwen_path = args.qwen3 or cached_file("aochongoliverli/math8k-sft-Qwen3-32B-16k-reasoning-traces", "data/train-00000-of-00001.parquet")
    inputs = [train_path, test_path, *omi_paths, qwen_path, args.sxiong, args.deepseek, args.extra_detailed]

    def traces():
        for path in omi_paths:
            print(f"Reading original-MATH OMI2 rows: {path.name}", flush=True)
            frame = pd.read_parquet(path, filters=[("problem_source", "=", "math")])
            yield from omi2_traces(frame, record_prefix=f"{path.name}:")
        print("Reading Qwen3", flush=True)
        yield from qwen3_traces(pd.read_parquet(qwen_path))
        print("Reading sxiong", flush=True)
        sxiong = pd.read_json(args.sxiong, lines=True) if args.sxiong.suffix == ".jsonl" else pd.read_parquet(args.sxiong)
        yield from sxiong_traces(sxiong)
        print("Reading local DeepSeek reasoning", flush=True)
        yield from deepseek_traces(pd.read_parquet(args.deepseek))
        print("Reading local extra-detailed written answers", flush=True)
        yield from deepseek_traces(pd.read_parquet(args.extra_detailed), extra_detailed=True)

    train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    frame, audit, aliases = build_mix(train, test, traces(), progress=True)
    sources = {"omi2_math", "qwen3_math8k", "sxiong_r1", "deepseek_regenerated", "deepseek_extra_detailed_written"}
    if set(frame.source) != sources:
        raise ValueError(f"Missing source(s): {sources - set(frame.source)}")
    source_stats = {}
    for source, group in frame.groupby("source"):
        lengths = group.reasoning_chars
        source_stats[source] = {
            "traces": len(group), "questions": group.question_id.nunique(),
            "mean_chars": float(lengths.mean()), "median_chars": float(lengths.median()),
            "max_chars": int(lengths.max()),
            "filter_counts": audit[audit.source == source].status.value_counts().to_dict(),
        }
    covered_ids = set(frame.question_id)
    manifest = {
        "max_reasoning_chars_exclusive": 10000, "truncation": False,
        "correctness": "source flags must be true where present; independently verify final answers against train gold",
        "question_ids": "original train IDs; first train row is canonical for whitespace-equivalent questions",
        "train_id_aliases": aliases, "decontamination": "exact or shared 13-word sequence with full MATH test",
        "rows": len(frame), "unique_questions": frame.question_id.nunique(),
        "mean_reasoning_chars": float(frame.reasoning_chars.mean()),
        "median_reasoning_chars": float(frame.reasoning_chars.median()),
        "max_reasoning_chars": int(frame.reasoning_chars.max()),
        "sources": source_stats,
        "train_questions_without_traces": [str(qid) for qid in train.question_id if str(qid) not in covered_ids],
        "input_files": [
            {"path": str(path.absolute()), "resolved_path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in inputs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    audit.to_parquet(args.output.with_suffix(".audit.parquet"), index=False)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"input_files", "train_questions_without_traces"}}, indent=2))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
