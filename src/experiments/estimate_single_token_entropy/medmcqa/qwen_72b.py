"""Proxy single-token entropy of Qwen/Qwen2.5-72B-Instruct on the 10k MedMCQA train sample.

Run from the repository root:
  PYTHONPATH=src python -m experiments.estimate_single_token_entropy.medmcqa.qwen_72b
"""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.medmcqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(model_id="Qwen/Qwen2.5-72B-Instruct", model_name="qwen_72b")
