"""Train llama_3b on the Llama-style MATH mix (Llama 70B + Llama 3B self + OMI2); 5 epochs, constant LR, 1k eval."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./reasoning_mix_under10k_llama_style/llama_3b_lora_large_constant",
        train_dataset="hendrycks_math_reasoning_mix_under10k_llama_style",
        save_schedule=[5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        eval_dataset="hendrycks_math_test_1k",
        lr_scheduler_type="constant",
    )
