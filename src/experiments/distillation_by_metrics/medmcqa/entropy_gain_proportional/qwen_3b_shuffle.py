"""Entropy-gain proportional sampling for qwen_3b on DeepSeek V3 MedMCQA traces; proxies Llama 70B + Qwen 72B.

Run from the repository root:
  PYTHONPATH=src python -m experiments.distillation_by_metrics.medmcqa.entropy_gain_proportional.qwen_3b_shuffle
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
        model_name="qwen_3b",
        relative_out_path="./entropy_gain_proportional/qwen_3b_shuffle",
        train_dataset="medmcqa_distilled_cot_deepseek_v3_correct",
        train_dataset_adapter=get_merged_adapter_with_data_mix(EntropyGainProportionalSampler),
        save_schedule=[10, 20, 50],
        complexity_estimator_override=SingleTokenEntropyWithRandomEstimator(),
        shuffle=True,
    )
