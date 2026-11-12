"""Proxy entropy for the qwen_3b student: Qwen2.5-32B on the unfiltered qwen_72b CoT pool."""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.hendrycks_math.shared import run


if __name__ == "__main__":
    freeze_support()
    run(
        model_id="Qwen/Qwen2.5-32B-Instruct",
        model_name="qwen_32b",
        train_dataset="hendrycks_math_cot_qwen_72b_all_under10k",
    )
