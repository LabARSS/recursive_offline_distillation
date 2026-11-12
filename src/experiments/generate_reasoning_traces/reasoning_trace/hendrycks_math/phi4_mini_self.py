"""Generate Phi-4-mini self-distillation traces with a single greedy pass.

Run from the repository root:
  PYTHONPATH=src python -m experiments.generate_reasoning_traces.reasoning_trace.hendrycks_math.phi4_mini_self

Outputs:
  data/out/distillation/hendrycks_math_self_distilled_phi4_mini_raw/
  data/out/distillation/hendrycks_math_self_distilled_phi4_mini.parquet
"""

from multiprocessing import freeze_support

from experiments.generate_reasoning_traces.reasoning_trace.hendrycks_math.self_distillation_shared import (
    run_greedy_self_distillation,
)


def main() -> None:
    run_greedy_self_distillation(model_id="microsoft/Phi-4-mini-instruct", model_name="phi4_mini")


if __name__ == "__main__":
    freeze_support()
    main()
