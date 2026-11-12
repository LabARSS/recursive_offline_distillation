"""Build a Qwen-style MATH mix: Qwen2.5-72B CoT, filtered Mistral Medium CoT, Qwen 3B self chains.

All chains are verified correct against train gold, strictly under 10,000
characters, test-decontaminated, and never truncated (same pipeline as the
CoT-teacher mix; teacher chains lose their trailing ``Answer: <x>`` line).

Mistral Medium is restricted to its Qwen-compatible chains. A chain is dropped
if it has any Markdown header line (``^#``), a line starting with
"Alright"/"Okay", the words "Wait"/"Hmm", or 5,000+ characters. This removes
its ``## Step`` layout and its reasoning-model voice (median ~5.3k chars).

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.build_hendrycks_math_reasoning_mix_10k_qwen_style
"""

import json
import re

import pandas as pd

from core.utils.cot_answers import verify_math_answer

from .build_hendrycks_math_reasoning_mix_10k import DISTILL, ROOT, Trace, build_mix, sha256_file
from .download_hendrycks_math import extract_final_answer
from .build_hendrycks_math_reasoning_mix_10k_cot_teachers import length_stats, teacher_traces

OUTPUT = DISTILL / "hendrycks_math_reasoning_mix_under10k_qwen_style.parquet"
SEED = 42
MISTRAL_MAX_CHARS = 5000
MISTRAL_EXCLUDE = {
    "markdown_header": re.compile(r"^#", re.M),
    "alright_okay_line": re.compile(r"^(?:Alright|Okay)", re.M),
    "wait_hmm": re.compile(r"\b(?:Wait|Hmm)\b"),
}


def self_traces(frame: pd.DataFrame, source: str = "self_qwen_3b"):
    for row in frame.to_dict("records"):
        yield Trace(
            source, f"{row['question_id']}:{row['pass']}", row["question"], row["distill_reasoning"],
            row["distill_answer"], "distill_reasoning", row["distill_ans_correct"] is True,
        )


def unwrap_math_delimiters(answer: str) -> str:
    """Strip one outer $...$, \\(...\\) or \\[...\\] that wraps the whole answer."""
    answer = answer.strip()
    for opening, closing in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if answer.startswith(opening) and answer.endswith(closing) and len(answer) > len(opening + closing):
            inner = answer[len(opening):-len(closing)].strip()
            # Skip answers like "$1$ and $2$" whose wrappers are not one pair.
            return answer if opening in inner or closing in inner else inner
    return answer


def clean_answer(answer: str) -> str:
    """Normalize the answer that follows </think>: keep a \\boxed{} answer's contents
    (dropping any prose around it), then strip one outer math-delimiter wrapper."""
    if "\\boxed" in answer or "\\fbox" in answer:
        try:
            answer = extract_final_answer(answer)
        except ValueError:
            pass
    return unwrap_math_delimiters(answer)


def assert_answers_verify(output: pd.DataFrame) -> None:
    failures = [a for a, g in zip(output.distill_answer, output.answer) if not verify_math_answer(str(g), a)]
    if failures:
        raise ValueError(f"{len(failures)} cleaned answers no longer verify, e.g. {failures[:3]}")


def mistral_exclusion(reasoning: str, chars: int) -> str | None:
    for name, pattern in MISTRAL_EXCLUDE.items():
        if pattern.search(reasoning):
            return name
    return "at_or_over_5000_chars" if chars >= MISTRAL_MAX_CHARS else None


def main() -> None:
    paths = {
        "train": ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet",
        "test": ROOT / "data/source/hendrycks_math/hendrycks_math_test.parquet",
        "qwen_72b": DISTILL / "hendrycks_math_distilled_cot_qwen_72b.parquet",
        "mistral_medium": DISTILL / "hendrycks_math_distilled_cot_mistral_medium.parquet",
        "self_qwen_3b": DISTILL / "hendrycks_math_self_distilled_qwen_3b.parquet",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}

    sources = {
        "qwen_72b": teacher_traces("qwen_72b", frames["qwen_72b"]),
        "mistral_medium": teacher_traces("mistral_medium", frames["mistral_medium"]),
        "self_qwen_3b": self_traces(frames["self_qwen_3b"]),
    }
    parts, audits = [], []
    for name, traces in sources.items():
        frame, audit, _ = build_mix(frames["train"], frames["test"], traces, max_chars=10000)
        if name == "mistral_medium":
            reasons = [mistral_exclusion(r, c) for r, c in zip(frame.distill_reasoning, frame.reasoning_chars)]
            excluded = dict(zip(frame.reasoning_sha256, reasons))
            audit["status"] = [
                excluded.get(sha) or status if status == "included" else status
                for status, sha in zip(audit.status, audit.reasoning_sha256)
            ]
            frame = frame[[r is None for r in reasons]]
        print(f"{name}: {audit.status.value_counts().to_dict()}", flush=True)
        parts.append(frame)
        audits.append(audit)
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError(f"Input changed during construction: {path}")

    output = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    output = output.drop_duplicates(["question_id", "reasoning_sha256"]).reset_index(drop=True)
    # The answer follows </think> and the prompt forbids math delimiters; some
    # self (and a few Qwen 72B) answers keep them, e.g. "\(\frac{400}{9}\)".
    output["distill_answer"] = output.distill_answer.map(clean_answer)
    assert_answers_verify(output)
    output["original_question_id"] = output.question_id
    output["question_id"] = [f"math_qwen_style_{i:06d}" for i in range(len(output))]
    assert output.trace_id.is_unique and output.distill_reasoning.str.len().lt(10000).all()
    assert not output.distill_reasoning.str.contains(r"\nAnswer:[^\n]*$", regex=True).any()
    mistral = output[output.source == "mistral_medium"]
    assert not any(p.search(r) for r in mistral.distill_reasoning for p in MISTRAL_EXCLUDE.values())
    assert mistral.reasoning_chars.lt(MISTRAL_MAX_CHARS).all()

    manifest = {
        "rows": len(output), "unique_questions": output.original_question_id.nunique(),
        **length_stats(output.reasoning_chars),
        "max_reasoning_chars_exclusive": 10000, "traces_truncated": 0,
        "mistral_filter": {
            "exclude_if_matches": {name: pattern.pattern for name, pattern in MISTRAL_EXCLUDE.items()},
            "max_chars_exclusive": MISTRAL_MAX_CHARS,
        },
        "correctness": "final-answer verification against train gold (self chains also require their source flag)",
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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)
    pd.concat(audits, ignore_index=True).to_parquet(OUTPUT.with_suffix(".audit.parquet"), index=False)
    manifest["output_sha256"] = sha256_file(OUTPUT)
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "unique_questions", "median_chars", "p95_chars", "sources", "chains_per_question")}, indent=2))
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
