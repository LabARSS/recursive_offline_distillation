"""Generate Qwen/Qwen2.5-3B-Instruct CoT replies on the gpqa train split and keep the correct ones for self-distillation.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.gpqa.base_model.qwen_3b_cot_train
"""

from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.gpqa.base_model.shared import (
    SELF_DISTILL_SPLIT,
    evaluate_base_model,
    export_self_distill_traces,
)

if __name__ == "__main__":
    freeze_support()
    evaluate_base_model(base_model_id="Qwen/Qwen2.5-3B-Instruct", model_name="qwen_3b", split=SELF_DISTILL_SPLIT)
    export_self_distill_traces(model_name="qwen_3b")
