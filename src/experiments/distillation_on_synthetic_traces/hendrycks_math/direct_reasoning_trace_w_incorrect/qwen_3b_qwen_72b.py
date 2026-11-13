"""Train qwen_3b on Qwen2.5-72B MATH CoT traces (under 10k chars, incl. incorrect, non-empty); constant LR 1e-5, 5 epochs, 1k eval.

Only epoch 5 is checkpointed, so only epoch 5 is evaluated.
Pool: preprocessing.build_hendrycks_math_cot_under10k --teachers qwen_72b --include-incorrect

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.hendrycks_math.direct_reasoning_trace_w_incorrect.qwen_3b_qwen_72b
"""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./direct_reasoning_trace_w_incorrect/qwen_3b_qwen_72b",
        train_dataset="hendrycks_math_distilled_cot_qwen_72b_under10k_w_incorrect",
        save_schedule=[5],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        eval_dataset="hendrycks_math_test_1k",
    )
