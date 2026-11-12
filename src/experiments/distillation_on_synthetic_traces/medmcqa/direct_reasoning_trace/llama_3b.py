"""Train llama_3b on DeepSeek V3 CoT traces (correct only) for the 10k MedMCQA train sample.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_on_synthetic_traces.medmcqa.direct_reasoning_trace.llama_3b
"""

from multiprocessing import freeze_support

from experiments.distillation_on_synthetic_traces.medmcqa.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./direct_reasoning_trace/llama_3b",
        train_dataset="medmcqa_distilled_cot_deepseek_v3_correct",
        save_schedule=[1, 2, 5, 10],
    )
