"""
Sample 10k MedMCQA train questions and 1k test questions in the GPQA parquet format,
so the GPQA multiple-choice dataset classes can be reused unchanged.

MedMCQA's own test split is unlabeled (cop = -1), so the 1k test set is drawn from
the validation split. Train questions whose text matches a sampled test question are
dropped.

Only the two needed parquet files are downloaded; the 183k-row train split is read as an
Arrow table and only the sampled rows are converted to pandas.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.download_medmcqa

Outputs:
  data/source/medmcqa/medmcqa_train_10k.parquet
  data/source/medmcqa/medmcqa_test_1k.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = "openlifescienceai/medmcqa"
TRAIN_FILE = "data/train-00000-of-00001.parquet"
VALIDATION_FILE = "data/validation-00000-of-00001.parquet"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data/source/medmcqa"
COLUMNS = ["id", "question", "opa", "opb", "opc", "opd", "cop", "subject_name", "choice_type"]
OPTION_LETTERS = "ABCD"
TRAIN_SIZE = 10_000
TEST_SIZE = 1_000
TRAIN_MARGIN = 100  # Extra draws to replace train rows that overlap the test sample.
SEED = 42


def sample_rows(filename: str, n: int, seed: int) -> pd.DataFrame:
    path = hf_hub_download(DATASET_ID, filename, repo_type="dataset")
    table = pq.read_table(path, columns=COLUMNS)
    indices = np.random.default_rng(seed).choice(table.num_rows, size=n, replace=False)
    print(f"Sampled {n} of {table.num_rows} rows from {filename}")
    return table.take(indices).to_pandas()


def question_key(question: str) -> str:
    return " ".join(question.split()).lower()


def to_gpqa_format(df: pd.DataFrame) -> pd.DataFrame:
    assert df["cop"].between(0, 3).all(), "Sampled rows must have a labeled correct option"
    return pd.DataFrame(
        {
            "question": df["question"],
            "options": [str([a, b, c, d]) for a, b, c, d in zip(df.opa, df.opb, df.opc, df.opd)],
            "answer": [OPTION_LETTERS[cop] for cop in df["cop"]],
            "answer_index": df["cop"],
            "question_id": df["id"].astype(str),
            "subject_name": df["subject_name"],
            "choice_type": df["choice_type"],
        }
    ).reset_index(drop=True)


def main():
    test = sample_rows(VALIDATION_FILE, TEST_SIZE, SEED)

    train = sample_rows(TRAIN_FILE, TRAIN_SIZE + TRAIN_MARGIN, SEED)
    test_questions = set(test["question"].map(question_key))
    overlap = train["question"].map(question_key).isin(test_questions)
    print(f"Dropped {overlap.sum()} train rows that overlap the test sample")
    train = train[~overlap]
    assert len(train) >= TRAIN_SIZE, f"Only {len(train)} train rows left after decontamination"
    train = train.head(TRAIN_SIZE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in (("medmcqa_train_10k", train), ("medmcqa_test_1k", test)):
        out = to_gpqa_format(df)
        assert out["question_id"].is_unique
        output_path = OUTPUT_DIR / f"{name}.parquet"
        out.to_parquet(output_path, index=False)
        print(f"Saved {len(out)} rows to {output_path}")
        print(f"Answer distribution: {out['answer'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
