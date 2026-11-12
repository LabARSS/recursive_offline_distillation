"""Proxy entropy for the Llama 3B student: Llama-3.1-8B on the Llama 70B CoT pool.

Llama 3 has no mid-size model comparable to Qwen2.5-32B. The 8B model keeps the
proxy distinct from the teacher and shares the 3B student's tokenizer.
"""

from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.hendrycks_math.shared import run


if __name__ == "__main__":
    freeze_support()
    run(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_name="llama_8b",
        train_dataset="hendrycks_math_cot_llama_70b_under10k",
    )
