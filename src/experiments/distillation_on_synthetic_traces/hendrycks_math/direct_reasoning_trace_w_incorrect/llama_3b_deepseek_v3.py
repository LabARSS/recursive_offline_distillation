"""Train llama_3b on DeepSeek V3 MATH CoT traces (under 10k chars, incl. incorrect, non-empty); constant LR 1e-5, 5 epochs, 1k eval.

Only epoch 5 is checkpointed, so only epoch 5 is evaluated.
Pool: preprocessing.build_hendrycks_math_cot_under10k --teachers deepseek_v3 --include-incorrect

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.hendrycks_math.direct_reasoning_trace_w_incorrect.llama_3b_deepseek_v3
"""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./direct_reasoning_trace_w_incorrect/llama_3b_deepseek_v3",
        train_dataset="hendrycks_math_distilled_cot_deepseek_v3_under10k_w_incorrect",
        save_schedule=[5, 10],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        eval_dataset="hendrycks_math_test_1k",
    )
