"""Entropy-gain proportional sampling for llama_3b on DeepSeek V3 MedMCQA traces; proxies Llama 70B + Qwen 72B.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_by_metrics.medmcqa.entropy_gain_proportional.llama_3b_shuffle
"""

from multiprocessing import freeze_support

from core.complexity_estimation.entropy.single_token_entropy_with_random_estimator import (
    SingleTokenEntropyWithRandomEstimator,
)
from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from experiments.distillation_by_metrics.medmcqa.shared import get_merged_adapter_with_data_mix, run

if __name__ == "__main__":
    freeze_support()
    run(
        model_name="llama_3b",
        relative_out_path="./entropy_gain_proportional/llama_3b_shuffle",
        train_dataset="medmcqa_distilled_cot_deepseek_v3_under10k",
        train_dataset_adapter=get_merged_adapter_with_data_mix(EntropyGainProportionalSampler),
        save_schedule=[20, 50, 80, 100],
        # Measure entropy every 10 epochs; redraw the weighted sample every epoch in between.
        resampling_schedule=list(range(0, 100, 10)),
        complexity_estimator_override=SingleTokenEntropyWithRandomEstimator(),
        shuffle=True,
    )
