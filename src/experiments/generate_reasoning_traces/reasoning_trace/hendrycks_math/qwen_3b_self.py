"""Generate Qwen2.5-3B self-distillation traces with a single greedy pass.

Run from the repository root:
  PYTHONPATH=src python -m experiments.generate_reasoning_traces.reasoning_trace.hendrycks_math.qwen_3b_self

Outputs:
  data/out/distillation/hendrycks_math_self_distilled_qwen_3b_raw/
  data/out/distillation/hendrycks_math_self_distilled_qwen_3b.parquet
"""

from multiprocessing import freeze_support

from experiments.generate_reasoning_traces.reasoning_trace.hendrycks_math.self_distillation_shared import (
    run_greedy_self_distillation,
)


def main() -> None:
    run_greedy_self_distillation(model_id="Qwen/Qwen2.5-3B-Instruct", model_name="qwen_3b")


if __name__ == "__main__":
    freeze_support()
    main()
