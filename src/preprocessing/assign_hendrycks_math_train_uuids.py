"""Replace MATH train question IDs with UUIDs and propagate them to the CoT distillation files.

Distilled rows are matched to train rows by exact question text (unique in train),
and their gold answers must agree. Idempotent: existing UUID IDs are kept.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.assign_hendrycks_math_train_uuids
"""

import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/source/hendrycks_math/hendrycks_math_train.parquet"
DISTILLED = [
    ROOT / "data/out/distillation" / f"hendrycks_math_distilled_cot_{teacher}.parquet"
    for teacher in ("llama_70b", "mistral_medium", "qwen_72b", "deepseek_v3")
]


def is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def main() -> None:
    train = pd.read_parquet(TRAIN)
    assert train.question.is_unique and train.question_id.is_unique
    if not train.question_id.map(is_uuid).all():
        train["question_id"] = [str(uuid.uuid4()) for _ in range(len(train))]
        assert train.question_id.is_unique
        train.to_parquet(TRAIN, index=False)
        print(f"Assigned {len(train)} UUIDs -> {TRAIN}")

    gold = train.set_index("question")
    for path in DISTILLED:
        frame = pd.read_parquet(path)
        missing = ~frame.question.isin(gold.index)
        if missing.any():
            raise ValueError(f"{int(missing.sum())} question(s) in {path} are not in train")
        if not frame.question.is_unique:
            raise ValueError(f"Duplicate questions in {path}")
        if frame.answer.tolist() != gold.loc[frame.question, "answer"].tolist():
            raise ValueError(f"Gold answers in {path} disagree with train")
        frame["question_id"] = gold.loc[frame.question, "question_id"].values
        frame.to_parquet(path, index=False)
        print(f"Matched {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
