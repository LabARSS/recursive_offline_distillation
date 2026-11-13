"""Train llama_3b on DeepSeek V3 MATH CoT traces (correct, under 10k chars); constant LR 1e-5, 1k eval.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.hendrycks_math.direct_reasoning_trace.llama_3b
"""

from multiprocessing import freeze_support

from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./direct_reasoning_trace/llama_3b",
        train_dataset="hendrycks_math_distilled_cot_deepseek_v3_under10k",
        save_schedule=[2, 5, 10],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        eval_dataset="hendrycks_math_test_1k",
    )
