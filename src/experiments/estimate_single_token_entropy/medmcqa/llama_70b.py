"""Proxy single-token entropy of meta-llama/Llama-3.3-70B-Instruct on the 10k MedMCQA train sample.

Run from the repository root:
  PYTHONPATH=src python -m experiments.estimate_single_token_entropy.medmcqa.llama_70b
"""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.medmcqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(model_id="meta-llama/Llama-3.3-70B-Instruct", model_name="llama_70b")
