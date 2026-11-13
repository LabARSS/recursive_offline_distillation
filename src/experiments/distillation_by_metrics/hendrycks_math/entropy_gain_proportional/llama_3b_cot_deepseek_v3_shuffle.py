"""Entropy-gain proportional sampling for llama_3b on DeepSeek V3 MATH CoT traces; proxies Llama 70B + Qwen 72B.

Proxy entropy is estimated once on hendrycks_math_train (UUID question IDs) and merged onto the
trace pool. Each epoch draws 10% of the pool, so epochs 20/50/100 match the compute of the
2/5/10-epoch full-pool baseline (distillation_on_synthetic_traces/hendrycks_math/direct_reasoning_trace).

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_by_metrics.hendrycks_math.entropy_gain_proportional.llama_3b_cot_deepseek_v3_shuffle
"""

from multiprocessing import freeze_support

from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.training.lora_trainer import LoRASpecificTrainingArgsLarge
from experiments.distillation_by_metrics.hendrycks_math.shared import run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./entropy_gain_proportional/llama_3b_cot_llama_70b_under10k_w_incorrect",
        train_dataset="hendrycks_math_distilled_cot_llama_70b_under10k_w_incorrect",
        sampler_cls=EntropyGainProportionalSampler,
        save_schedule=[20, 50, 80, 100],
        max_thinking_tokens=4096,
        lora_training_args=LoRASpecificTrainingArgsLarge(),
        learning_rate=1e-5,
        shuffle=True,
        proxy_models=("llama_70b",),
        proxy_entropy_dataset="hendrycks_math_train",
        eval_dataset="hendrycks_math_test_1k",
        empty_thinking=True,
    )
