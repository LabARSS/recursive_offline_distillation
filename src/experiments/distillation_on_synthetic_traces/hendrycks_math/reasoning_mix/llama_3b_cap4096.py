from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="llama_3b",
    relative_out_path="./reasoning_mix/llama_3b",
    train_dataset="hendrycks_math_reasoning_mix_llama_3b",
    save_schedule=[1, 2, 5],
    max_thinking_tokens=4096,
)
