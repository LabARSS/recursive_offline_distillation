"""Proxy entropy for the llama_3b student: Llama-3.1-8B on the unfiltered llama_70b CoT pool."""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.hendrycks_math.shared import run


if __name__ == "__main__":
    freeze_support()
    run(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_name="llama_8b",
        train_dataset="hendrycks_math_cot_llama_70b_all_under10k",
    )
