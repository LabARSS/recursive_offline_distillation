from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="phi4_mini",
    relative_out_path="./openmathinstruct2_20k/phi4_mini",
    train_dataset="hendrycks_math_openmathinstruct2_math_all2sol_aug7000",
    save_schedule=[1, 2, 5],
    max_thinking_tokens=4096,
)
