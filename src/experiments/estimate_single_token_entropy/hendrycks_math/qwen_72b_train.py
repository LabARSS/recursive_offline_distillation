"""Proxy single-token entropy of Qwen/Qwen2.5-72B-Instruct on the full MATH train set (UUID question IDs).

Run from the repository root:
  PYTHONPATH=src python -m experiments.estimate_single_token_entropy.hendrycks_math.qwen_72b_train
"""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        model_name="qwen_72b",
        train_dataset="hendrycks_math_train",
        data_dir="data/source/hendrycks_math",
    )
