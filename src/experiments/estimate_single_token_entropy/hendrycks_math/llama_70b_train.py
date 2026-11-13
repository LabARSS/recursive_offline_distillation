"""Proxy single-token entropy of meta-llama/Llama-3.3-70B-Instruct on the full MATH train set (UUID question IDs).

Run from the repository root:
  PYTHONPATH=src python -m experiments.estimate_single_token_entropy.hendrycks_math.llama_70b_train
"""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        model_name="llama_70b",
        train_dataset="hendrycks_math_train",
        data_dir="data/source/hendrycks_math",
    )
