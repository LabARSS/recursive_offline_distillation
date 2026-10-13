"""
Download all Hendrycks MATH subjects from HuggingFace and save both splits
as parquet in the same format as the GSM8K source artifacts.

Usage:
  python src/preprocessing/download_hendrycks_math.py

Outputs:
  data/source/hendrycks_math/hendrycks_math_train.parquet
  data/source/hendrycks_math/hendrycks_math_test.parquet

Two training solutions have empty boxes despite explaining that the answer is
zero. Their answers are corrected to "0"; answer_w_steps preserves the source.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_ID = "EleutherAI/hendrycks_math"
DATASET_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
SPLITS = ("train", "test")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/source/hendrycks_math"
COLUMNS = ("question", "answer", "answer_w_steps", "question_id")

# These two Number Theory solutions omit zero from their boxes. Match their
# complete text so other missing or malformed source answers still raise.
SOURCE_ANSWER_CORRECTIONS = {
    (
        r"Observe that for all $k \in 1< k< n$, since $k$ divides $n!$, "
        r"$k$ also divides $n!+k$. Therefore, all numbers $a$ in the range "
        r"$n!+1<a<n!+n$ are composite. Therefore there are $\boxed{}$ primes "
        r"in that range."
    ): "0",
    (
        r"First, note that $n$ does not have a prime number larger than $61$ "
        r"as one of its factors. Also, note that $n$ does not equal $1$."
        "\n"
        r"Therefore, since the prime factorization of $n$ only has primes "
        r"from $2$ to $59$, $n$ and $P$ share at least one common factor other "
        r"than $1$. Therefore $P+n$ is not prime for any $n$, so the answer is "
        r"$\boxed{}$."
    ): "0",
}


def extract_final_answer(solution: str) -> str:
    """Extract the last boxed answer, preserving the LaTeX inside it."""
    command = r"\boxed"
    index = solution.rfind(command)
    if index < 0:
        command = r"\fbox"
        index = solution.rfind(command)
    if index < 0:
        raise ValueError("Solution contains no boxed answer")

    start = index + len(command)
    while start < len(solution) and solution[start].isspace():
        start += 1
    if start == len(solution):
        raise ValueError("Box command has no answer")

    if solution[start] == "{":
        depth = 0
        escaped = False
        for end in range(start, len(solution)):
            char = solution[end]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answer = solution[start + 1 : end].strip()
                    break
        else:
            raise ValueError("Boxed answer has unbalanced braces")
    else:
        end = solution.find("$", start)
        if end < 0:
            raise ValueError("Unbraced boxed answer has no closing '$'")
        answer = solution[start:end].strip()

    if not answer:
        raise ValueError("Boxed answer is empty")
    return answer


def main():
    subjects = {}
    for config in DATASET_CONFIGS:
        ds = load_dataset(DATASET_ID, config)
        subjects[config] = ds
        for split in SPLITS:
            print(f"Downloaded {len(ds[split])} rows from {DATASET_ID}/{config}/{split}")

    for split in SPLITS:
        rows = []
        for config in DATASET_CONFIGS:
            for idx, example in enumerate(subjects[config][split]):
                solution = example["solution"]
                try:
                    if (
                        split == "train"
                        and config == "number_theory"
                        and solution in SOURCE_ANSWER_CORRECTIONS
                    ):
                        answer = SOURCE_ANSWER_CORRECTIONS[solution]
                        print(f"Corrected empty answer to {answer} for {split}/{config} row {idx}")
                    else:
                        answer = extract_final_answer(solution)
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot extract answer from {split}/{config} row {idx}: {exc}"
                    ) from exc

                rows.append(
                    {
                        "question": example["problem"],
                        "answer": answer,
                        "answer_w_steps": solution,
                        "question_id": str(len(rows)),
                    }
                )

        df = pd.DataFrame(rows, columns=COLUMNS)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"hendrycks_math_{split}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"Saved {len(df)} rows to {output_path}")
        print(f"Columns: {list(df.columns)}")
        print(df.head(2).to_string())


if __name__ == "__main__":
    main()
