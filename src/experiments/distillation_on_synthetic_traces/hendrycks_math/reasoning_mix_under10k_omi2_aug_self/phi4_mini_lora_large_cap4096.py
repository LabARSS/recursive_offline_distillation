"""Train on the extended MATH mix for five epochs; evaluate only epoch five."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="phi4_mini",
        relative_out_path="./reasoning_mix_under10k_omi2_aug_self/phi4_mini_lora_large",
        train_dataset="hendrycks_math_reasoning_mix_under10k_omi2_aug_self",
        save_schedule=[2, 5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
    )
