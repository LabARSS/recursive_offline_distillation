from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="llama_3b",
    relative_out_path="./direct_reasoning_trace/llama_3b_head_truncated8192",
    train_dataset="hendrycks_math_distilled_detailed_from_answer",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
)
