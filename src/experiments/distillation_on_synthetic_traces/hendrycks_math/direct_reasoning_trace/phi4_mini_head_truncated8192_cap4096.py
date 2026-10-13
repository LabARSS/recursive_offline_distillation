from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="phi4_mini",
    relative_out_path="./direct_reasoning_trace/phi4_mini_head_truncated8192",
    train_dataset="hendrycks_math_distilled_deepseek_v4_flash_regenerate_incorrect_w_large_head8192",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
)
