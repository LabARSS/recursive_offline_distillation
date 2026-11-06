"""Train on the same-family CoT-teacher MATH mix (Llama 70B, Qwen 72B, 50% DeepSeek V3, 25% OMI2)."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./reasoning_mix_under10k_cot_teachers/llama_3b_lora_large",
        train_dataset="hendrycks_math_reasoning_mix_under10k_cot_teachers",
        save_schedule=[2, 5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        # dataset_cls=HendrycksMathReasoningResponseDatasetDetailedPrompt,
    )
