"""Train qwen_3b only on same-family Qwen 72B CoT chains; 5 epochs, constant LR, 1k eval."""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./cot_teacher_qwen_72b/qwen_3b_lora_large_constant",
        train_dataset="hendrycks_math_cot_qwen_72b_under10k",
        save_schedule=[5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        eval_dataset="hendrycks_math_test_1k",
        lr_scheduler_type="constant",
    )
