from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="llama_3b",
    relative_out_path="./openmathinstruct2/llama_3b",
    train_dataset="hendrycks_math_openmathinstruct2_math5000_aug5000",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
)
