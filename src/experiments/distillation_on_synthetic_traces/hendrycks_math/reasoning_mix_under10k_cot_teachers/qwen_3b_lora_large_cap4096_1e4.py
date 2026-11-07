"""Train on the same-family CoT-teacher MATH mix (Llama 70B, Qwen 72B, 50% DeepSeek V3, 25% OMI2)."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./reasoning_mix_under10k_cot_teachers/qwen_3b_lora_large_1e4",
        train_dataset="hendrycks_math_reasoning_mix_under10k_cot_teachers",
        save_schedule=[5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-4,
        eval_dataset="hendrycks_math_test_1k",
        lr_scheduler_type="constant",
        # dataset_cls=HendrycksMathReasoningResponseDatasetDetailedPrompt,
    )
