"""Evaluate Qwen/Qwen2.5-3B-Instruct with CoT prompts on the 1,000-question hendrycks_math_test_1k sample.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.hendrycks_math.base_model.qwen_3b_cot_test1k
"""

from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.hendrycks_math.base_model.shared import evaluate_base_model

if __name__ == "__main__":
    freeze_support()
    evaluate_base_model(base_model_id="Qwen/Qwen2.5-3B-Instruct", model_name="qwen_3b", eval_dataset="hendrycks_math_test_1k")
