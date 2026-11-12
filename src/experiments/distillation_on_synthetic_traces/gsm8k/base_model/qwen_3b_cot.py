"""Evaluate Qwen/Qwen2.5-3B-Instruct with CoT prompts on gsm8k_test.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.gsm8k.base_model.qwen_3b_cot
"""

from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.gsm8k.base_model.shared import evaluate_base_model

if __name__ == "__main__":
    freeze_support()
    evaluate_base_model(base_model_id="Qwen/Qwen2.5-3B-Instruct", model_name="qwen_3b")
