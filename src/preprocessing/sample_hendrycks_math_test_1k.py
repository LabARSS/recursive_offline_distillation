"""Sample 1,000 MATH test questions (seed 42) for faster evaluation.

Rows keep their original order and question_id (the evaluation row_id), so
results can be matched to full-test evaluations.

Run from the repository root:
  PYTHONPATH=src python -m preprocessing.sample_hendrycks_math_test_1k
"""

from pathlib import Path

import pandas as pd

SOURCE = Path(__file__).resolve().parents[2] / "data/source/hendrycks_math"

if __name__ == "__main__":
    test = pd.read_parquet(SOURCE / "hendrycks_math_test.parquet")
    sample = test.sample(n=1000, random_state=42).sort_index()
    sample.reset_index(drop=True).to_parquet(SOURCE / "hendrycks_math_test_1k.parquet", index=False)
    print(f"Saved {len(sample)} rows to {SOURCE / 'hendrycks_math_test_1k.parquet'}")
