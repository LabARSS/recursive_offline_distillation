from multiprocessing import freeze_support

from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.datasets.hendrycks_math.hendrycks_math_reasoning_response_dataset_detailed_prompt import (
    HendrycksMathReasoningResponseDatasetDetailedPrompt,
)
from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_by_metrics.hendrycks_math.shared import run


if __name__ == "__main__":
    freeze_support()
    run(
        model_name="qwen_3b",
        relative_out_path="./entropy_gain_proportional/qwen_3b_lora_large_shuffle",
        train_dataset="hendrycks_math_reasoning_mix_under10k_omi2_aug_self",
        sampler_cls=EntropyGainProportionalSampler,
        save_schedule=[10, 20, 50, 80, 100, 150, 200],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        shuffle=True,
        dataset_cls=HendrycksMathReasoningResponseDatasetDetailedPrompt,
    )
