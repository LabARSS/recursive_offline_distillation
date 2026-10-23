from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_on_synthetic_traces.hendrycks_math.shared import run

run(
    model_name="qwen_3b",
    relative_out_path="./openmathinstruct2/qwen_3b_lora_large",
    train_dataset="hendrycks_math_openmathinstruct2_math5000_aug5000",
    save_schedule=[2, 5, 10, 15, 20],
    max_thinking_tokens=4096,
    lora_training_args=LoRASpecificTrainingArgsLarge(),
)
