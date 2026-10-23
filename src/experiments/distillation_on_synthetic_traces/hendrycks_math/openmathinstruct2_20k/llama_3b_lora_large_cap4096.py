from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="llama_3b",
    relative_out_path="./openmathinstruct2_20k/llama_3b_lora_large",
    train_dataset="hendrycks_math_openmathinstruct2_math_all2sol_aug7000",
    save_schedule=[1, 2, 5],
    max_thinking_tokens=4096,
    lora_training_args=LoRASpecificTrainingArgsLarge(),
)
