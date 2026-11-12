"""Rewrite Llama-70B MATH CoT final lines into the required ``Answer: <x>`` format.

Llama-3.3-70B ignored the prompt's final-line format and ended with
``The final answer is[:] $\\boxed{x}$`` (and variants). Only that last line is
rewritten; the chain body is untouched. Correctness is then re-graded with the
same extractor/verifier used during distillation. Idempotent; edits in place.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.fix_llama_70b_cot_answer_format
"""

import re
from pathlib import Path

import pandas as pd

from core.utils.cot_answers import extract_math_answer, verify_math_answer

PATH = Path(__file__).resolve().parents[2] / "data/out/distillation/hendrycks_math_distilled_cot_llama_70b.parquet"
PREFIX = re.compile(r"^(?:#+\s*)?(?:The final answer is:?|Answer:)\s*", re.IGNORECASE)


def unwrap_boxed(text: str) -> str:
    match = re.fullmatch(r"\\boxed\s*\{(.*)\}", text, re.DOTALL)
    if not match:
        return text
    inner, depth = match.group(1), 0
    for char in inner:  # Only unwrap when the braces enclose the whole answer.
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth < 0:
            return text
    return inner.strip() if depth == 0 else text


def normalize_answer(line: str) -> str:
    answer = line.strip()
    while (stripped := PREFIX.sub("", answer, count=1)) != answer:
        answer = stripped
    answer = answer.rstrip(".").strip()
    if len(answer) >= 2 and answer.startswith("$") and answer.endswith("$") and not answer.startswith("$$"):
        answer = answer[1:-1].strip()
    return unwrap_boxed(answer).rstrip(".").strip()


def fix_response(response: str) -> str:
    if not isinstance(response, str) or not response.strip() or extract_math_answer(response, ()):
        return response
    lines = response.rstrip().splitlines()
    nonempty = [i for i in range(len(lines)) if lines[i].strip()]
    last = nonempty[-1]
    # Variant: "The final answer is:" alone, with the answer on the next line.
    if len(nonempty) > 1 and re.fullmatch(r"The final answer is:?", lines[nonempty[-2]].strip(), re.IGNORECASE):
        lines[nonempty[-2]:] = [f"The final answer is: {lines[last].strip()}"]
        last = nonempty[-2]
    if not re.match(r"(?:#+\s*)?The final answer is", lines[last].strip(), re.IGNORECASE):
        return response
    answer = normalize_answer(lines[last])
    if not answer:
        return response
    return "\n".join([*lines[:last], f"Answer: {answer}"])


def main() -> None:
    df = pd.read_parquet(PATH)
    before = df.distill_ans_correct.copy()
    fixed = df.distill_answer.map(fix_response)
    changed = fixed.ne(df.distill_answer) & df.distill_answer.notna()
    df["distill_answer"] = fixed
    df["distill_ans_correct"] = [
        bool(isinstance(r, str) and (a := extract_math_answer(r, ())) and verify_math_answer(str(g), a))
        for r, g in zip(df.distill_answer, df.answer)
    ]
    print(f"Rewrote final line of {changed.sum():,}/{len(df):,} responses")
    print(f"Still non-conforming: {df.distill_answer.map(lambda r: not (isinstance(r, str) and extract_math_answer(r, ()))).sum():,}")
    print(f"Correct: {before.sum():,} -> {df.distill_ans_correct.sum():,}")
    df.to_parquet(PATH, index=False)
    print(f"Saved {PATH}")


if __name__ == "__main__":
    main()
