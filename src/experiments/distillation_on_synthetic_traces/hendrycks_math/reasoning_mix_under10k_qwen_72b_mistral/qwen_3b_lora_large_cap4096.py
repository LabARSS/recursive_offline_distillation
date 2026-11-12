"""Train qwen_3b on Qwen2.5-72B + filtered Mistral Medium CoT chains (correct, under 10k chars); constant LR 1e-5, 1k eval."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./reasoning_mix_under10k_qwen_72b_mistral/qwen_3b_lora_large",
        train_dataset="hendrycks_math_reasoning_mix_under10k_qwen_72b_mistral",
        save_schedule=[5, 10],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        eval_dataset="hendrycks_math_test_1k",
    )
