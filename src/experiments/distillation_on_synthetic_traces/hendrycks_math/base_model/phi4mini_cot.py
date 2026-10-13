"""Evaluate microsoft/Phi-4-mini-instruct with CoT prompts on hendrycks_math_test.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.hendrycks_math.base_model.phi4mini_cot
"""

from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.hendrycks_math.base_model.shared import evaluate_base_model

if __name__ == "__main__":
    freeze_support()
    evaluate_base_model(base_model_id="microsoft/Phi-4-mini-instruct", model_name="phi4_mini")
