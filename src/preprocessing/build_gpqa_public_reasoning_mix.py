"""Build a training-only GPQA mixture from separately downloaded solution sources.

Run after download_gpqa_public_traces.py:
  .venv/bin/python src/preprocessing/build_gpqa_public_reasoning_mix.py

External option orders are preserved, since written explanations reference letters.
original_question_id links every unique training example to train_original.parquet.
No reasoning is truncated. The character budget includes both thinking markers and
the final answer letter. The original train/test and corrected files are read only.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data/source/gpqa_reasoning_sources"
SPLITS = ROOT / "data/out/splits/random/gpqa"
OUTPUT = SPLITS / "train_reasoning_mix_public.parquet"
BOXED_LETTER = re.compile(r"\\boxed\s*\{\s*(?:\\text\s*\{\s*)?\(?([A-Da-d])\)?\s*\}?\s*\}")
CONTROL = re.compile(r"</?think>|<\|[^>]*\|>")


def normalize(text: str) -> str:
    # Preserve scientific symbol case (L and l, for example, can be different
    # variables) and compatibility characters such as superscript exponents.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def option_list(value: str | list) -> list[str]:
    options = ast.literal_eval(value) if isinstance(value, str) else list(value)
    if len(options) != 4 or not all(isinstance(x, str) for x in options):
        raise ValueError("Expected four string options")
    return options


def option_text(options: list[str], letter: str) -> str:
    letter = str(letter).strip().lower()
    if letter not in "abcd" or len(letter) != 1:
        raise ValueError("Invalid option letter")
    return options[ord(letter) - ord("a")]


def parse_sxiong_question(text: str) -> tuple[str, list[str]]:
    stem, choices = text.rsplit("Choices:", 1)
    stem = re.sub(r"^\s*Q:\s*", "", stem).strip()
    markers = list(re.finditer(r"(?m)^[ \t]*([A-D])[.)][ \t]+", choices))
    if "".join(m[1] for m in markers) != "ABCD":
        raise ValueError("Cannot unambiguously parse external options")
    options = [choices[m.end():markers[i + 1].start() if i < 3 else len(choices)].strip()
        for i, m in enumerate(markers)]
    return stem, option_list(options)


def parse_original_answer(text: str) -> tuple[str, str]:
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*#{1,6}\s*(.+?)\s*", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Cannot parse original explanation and answer")
    return match[1].strip(), match[2].strip()


def load_original_explanations(directory: Path) -> pd.DataFrame:
    official = directory / "original/gpqa_extended.csv"
    manifest = json.loads((directory / "download_manifest.json").read_text())
    provenance = manifest["sources"]["original"]
    if not provenance["official_requires_authentication"]:
        df = pd.read_csv(official).fillna("")
        rows = [{"source_record_id": str(r.get("Record ID", i)), "question": r["Question"],
            "explanation": r["Explanation"], "correct_answer_text": r["Correct Answer"]}
            for i, r in enumerate(df.to_dict("records"))]
    else:
        df = pd.read_parquet(directory / "original/gpqa_extended_public.parquet")
        rows = []
        for i, r in enumerate(df.to_dict("records")):
            explanation, answer = parse_original_answer(r["answer"])
            rows.append({"source_record_id": str(i), "question": r["question"],
                "explanation": explanation, "correct_answer_text": answer})
    out = pd.DataFrame(rows)
    out.to_parquet(directory / "original_explanations.parquet", index=False)
    return out


@dataclass
class Trace:
    source: str
    variant: str
    record_id: str
    question: str
    reasoning: str
    gold_text: str
    options: list[str] | None = None
    source_correct: bool = True
    source_answer: str = ""
    written_answer: str = ""
    split: str = ""
    model: str = ""
    verification: str = ""


def collect_traces(corrected: pd.DataFrame, original: pd.DataFrame,
                   sxiong: pd.DataFrame, spawn: pd.DataFrame) -> list[Trace]:
    traces = []
    for r in corrected.to_dict("records"):
        options = option_list(r["options"])
        traces.append(Trace("current_corrected", "corrected_concat", str(r["question_id"]),
            r["question"], r["distill_reasoning"], option_text(options, r["answer"]), options,
            source_correct=bool(r["distill_ans_correct"]), source_answer=r["distill_answer"],
            written_answer=r["distill_answer"], split="our_train", verification="exact_final_letter"))
    for r in original.to_dict("records"):
        traces.append(Trace("original_explanations", "expert_explanation", r["source_record_id"],
            r["question"], r["explanation"], r["correct_answer_text"],
            split="extended", model="question_writer", verification="exact_correct_option_text"))
    for r in sxiong.to_dict("records"):
        question, options = parse_sxiong_question(r["question"])
        predictions = BOXED_LETTER.findall(str(r["content"]))
        traces.append(Trace("sxiong_written", "written_solution", r["id"], question,
            r["content"] or "", option_text(options, r["answer"]), options,
            source_correct=r["correct"] is True, source_answer=r["answer"],
            written_answer=predictions[-1] if predictions else "", split=r["source_split"],
            model=r["model"], verification="source_grade_and_final_boxed_letter"))
    variants = (
        ("deepseek_r1_reasoning", "deepseek_reasoner", ("deepseek_reasoner", "reasoning")),
        ("deepseek_r1_written", "deepseek_reasoner", ("deepseek_reasoner", "full_response")),
        ("claude_standalone", "claude_sonnet_standalone", ("claude_sonnet", "standalone", "full_response")),
        ("claude_with_reasoning", "claude_sonnet_with_reasoning", ("claude_sonnet", "with_reasoning", "full_response")),
    )
    for r in spawn.to_dict("records"):
        for variant, prefix, keys in variants:
            value = r["model_responses"]
            for key in keys:
                value = value[key]
            traces.append(Trace("spawn99", variant, r["record_id"], r["question"], value or "",
                r["correct_answer"], source_correct=r[f"{prefix}_grade"] is True,
                source_answer=r[f"{prefix}_answer"], split="diamond", model=prefix,
                verification="source_grade_and_exact_correct_option_text"))
    return traces


def build_mix(train: pd.DataFrame, test: pd.DataFrame, traces: list[Trace],
              max_chars: int = 10000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["question_id"] = train.question_id.astype(str)
    test["question_id"] = test.question_id.astype(str)
    train_keys = train.question.map(normalize)
    test_keys = set(test.question.map(normalize))
    if train_keys.duplicated().any() or set(train_keys) & test_keys:
        raise ValueError("Ambiguous training questions or train/test question overlap")
    if set(train.question_id) & set(test.question_id):
        raise ValueError("Train/test ID overlap")
    lookup = {normalize(r["question"]): r for r in train.to_dict("records")}
    kept, audit = [], []
    seen = set()
    for trace in traces:
        record = {"source": trace.source, "source_variant": trace.variant,
            "source_record_id": trace.record_id, "source_split": trace.split,
            "original_question_id": "", "status": ""}
        key = normalize(trace.question)
        reason = ""
        target = lookup.get(key)
        if key in test_keys:
            reason = "test_question"
        elif target is None:
            reason = "not_in_training_questions"
        else:
            record["original_question_id"] = target["question_id"]
            local_options = option_list(target["options"])
            local_gold = option_text(local_options, target["answer"])
            options = trace.options if trace.options is not None else local_options
            if normalize(trace.gold_text) != normalize(local_gold):
                reason = "source_gold_disagrees_with_our_gold"
            elif sorted(map(normalize, options)) != sorted(map(normalize, local_options)):
                reason = "source_option_set_differs"
            elif not trace.source_correct:
                reason = "source_marked_incorrect"
            elif trace.options is not None and (
                len(trace.source_answer.strip()) != 1 or trace.source_answer.strip().lower() not in "abcd"
            ):
                reason = "invalid_source_answer_letter"
            elif trace.options is not None and normalize(option_text(options, trace.source_answer)) != normalize(local_gold):
                reason = "predicted_option_disagrees_with_our_gold"
            elif trace.options is not None and trace.written_answer.strip().lower() != trace.source_answer.strip().lower():
                reason = "written_answer_disagrees_or_missing"
            elif not trace.reasoning.strip():
                reason = "empty_reasoning"
            elif CONTROL.search(trace.reasoning):
                reason = "embedded_chat_or_thinking_control_token"
            else:
                # Some source questions contain duplicate option text. Preserve
                # the source's actual correct letter rather than choosing the
                # first equal string and silently changing the label.
                answer = trace.source_answer.strip().lower() if trace.options is not None else str(target["answer"]).lower()
                reasoning = trace.reasoning.strip()
                response_chars = len(f"<think>{reasoning}</think>{answer}")
                record["response_chars"] = response_chars
                signature = (key, tuple(map(normalize, options)), reasoning, answer)
                if response_chars > max_chars:
                    reason = "over_character_budget"
                elif signature in seen:
                    reason = "duplicate_training_target"
                else:
                    seen.add(signature)
                    kept.append({"question": target["question"], "options": str(options),
                        "answer": answer.upper(), "answer_index": ord(answer) - ord("a"),
                        "original_question_id": target["question_id"], "original_answer": target["answer"],
                        "distill_ans_correct": True, "distill_reasoning": reasoning, "distill_answer": answer,
                        "source": trace.source, "source_variant": trace.variant,
                        "source_record_id": trace.record_id, "source_split": trace.split,
                        "source_model": trace.model, "source_answer": trace.source_answer,
                        "answer_verification": trace.verification,
                        "option_order_changed": list(map(normalize, options)) != list(map(normalize, local_options)),
                        "response_chars": response_chars})
        record["status"] = reason or "included"
        audit.append(record)
    if not kept:
        raise ValueError("No eligible training traces")
    out = pd.DataFrame(kept).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["question_id"] = [f"gpqa_public_mix_{i:05d}" for i in range(len(out))]
    assert out.question_id.is_unique
    assert out.original_question_id.isin(train.question_id).all()
    assert not out.original_question_id.isin(test.question_id).any()
    assert not out.question.map(normalize).isin(test_keys).any()
    assert out.response_chars.le(max_chars).all()
    return out, pd.DataFrame(audit)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    original = load_original_explanations(SOURCES)
    corrected_path = SPLITS / "train_corrected_answer_deepseek_v4_pro_concat.parquet"
    paths = {"train": SPLITS / "train_original.parquet", "test": SPLITS / "test.parquet",
        "current_corrected": corrected_path, "original": SOURCES / "original_explanations.parquet",
        "sxiong": SOURCES / "sxiong_written_solutions.parquet", "spawn99": SOURCES / "spawn99_solutions.parquet"}
    data = {key: pd.read_parquet(path) for key, path in paths.items()}
    traces = collect_traces(data["current_corrected"], original, data["sxiong"], data["spawn99"])
    out, audit = build_mix(data["train"], data["test"], traces, args.max_chars, args.seed)
    out.to_parquet(OUTPUT, index=False)
    per_source = SPLITS / "public_reasoning_sources"
    per_source.mkdir(exist_ok=True)
    for source, rows in out.groupby("source"):
        rows.to_parquet(per_source / f"train_{source}.parquet", index=False)
    audit.to_csv(OUTPUT.with_suffix(".audit.csv"), index=False)
    counts = out.groupby(["source", "source_variant"]).agg(
        rows=("question_id", "size"), questions=("original_question_id", "nunique"),
        median_response_chars=("response_chars", "median"), max_response_chars=("response_chars", "max"))
    source_summary = counts.reset_index().to_dict("records")
    manifest = {"output": str(OUTPUT.relative_to(ROOT)), "rows": len(out),
        "unique_training_questions": out.original_question_id.nunique(), "train_questions": len(data["train"]),
        "test_questions": len(data["test"]), "test_question_overlap": 0, "test_id_overlap": 0,
        "max_response_chars": int(out.response_chars.max()), "character_budget": args.max_chars,
        "budget_definition": "len('<think>' + distill_reasoning.strip() + '</think>' + distill_answer.strip())",
        "traces_truncated": 0, "seed": args.seed, "sources": source_summary,
        "candidate_outcomes": audit.groupby(["source", "status"]).size().reset_index(name="rows").to_dict("records"),
        "input_files": {key: {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)} for key, path in paths.items()},
        "output_sha256": file_hash(OUTPUT),
        "notes": ["Unique question_id identifies a training trace; original_question_id is the unchanged training question ID.",
            "External option orders are retained to preserve option references in written solutions.",
            "Source correctness flags and exact answer alignment do not certify every reasoning step.",
            "Raw downloaded sources can contain held-out questions; only the matched train_* files and final mix are training data."]}
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("output", "rows", "unique_training_questions",
        "test_question_overlap", "max_response_chars", "traces_truncated", "sources", "candidate_outcomes")}, indent=2))


if __name__ == "__main__":
    main()
