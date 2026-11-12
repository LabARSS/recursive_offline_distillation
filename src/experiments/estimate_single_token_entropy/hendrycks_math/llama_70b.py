from multiprocessing import freeze_support

from experiments.estimate_single_token_entropy.hendrycks_math.shared import run


if __name__ == "__main__":
    freeze_support()
    run(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        model_name="llama_70b",
        train_dataset="hendrycks_math_reasoning_mix_under10k_omi2_aug_self",
    )
