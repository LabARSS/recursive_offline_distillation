"""Floored entropy-gain proportional sampling for llama_3b on the Llama-3.3-70B + OpenMathInstruct-2 MATH mix; proxy Llama-3.1-8B.

Weight per row: floor + max(student - proxy, 0), floor = mean clamped gap / 9
(10% of the draw weight is uniform), so rows at or below
the proxy stay drawable instead of being dropped. Entropy is measured every 10 epochs and
the sample is redrawn every epoch. Each epoch draws 10% of the
pool, so epoch 50 matches the compute of the 5-epoch full-pool baseline
(reasoning_mix_under10k_llama_70b_omi2/llama_3b_lora_large).
"""

from multiprocessing import freeze_support

from core.dataset_samplers.floored_entropy_gain_proportional_sampler import FlooredEntropyGainProportionalSampler
from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_by_metrics.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./entropy_gain_proportional/llama_3b_mix_llama_70b_omi2_floored_lora_large",
        train_dataset="hendrycks_math_reasoning_mix_under10k_llama_70b_omi2",
        sampler_cls=FlooredEntropyGainProportionalSampler,
        save_schedule=[50, 80, 100],
        # Measure entropy every 10 epochs; redraw the weighted sample every epoch in between.
        resampling_schedule=list(range(0, 100, 10)),
        redraw_each_epoch=True,
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        shuffle=True,
        proxy_models=("llama_8b",),
        eval_dataset="hendrycks_math_test_1k",
    )
