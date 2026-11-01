"""Train on the five-source mix for five epochs; evaluate only epoch five."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run


if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./reasoning_mix_under10k/qwen_3b_lora_large_constant",
        train_dataset="hendrycks_math_reasoning_mix_omi2_qwen3_sxiong_deepseek_under10k",
        save_schedule=[5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=5e-5,
        lr_scheduler_type="constant",
    )
