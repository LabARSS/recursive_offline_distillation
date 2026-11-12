"""MATH entropy-gain sampling budgets and reasoning/direct-answer supervision."""

from random import Random

import pandas as pd
import pytest

from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.datasets.hendrycks_math.hendrycks_math_reasoning_response_dataset_detailed_prompt import (
    HendrycksMathReasoningResponseDatasetDetailedPrompt,
)
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from experiments.distillation_by_metrics.hendrycks_math.shared import get_merged_adapter_with_data_mix


@pytest.mark.parametrize("dataset_size", [7500, 10000, 54805])
def test_sample_budgets_scale_with_the_training_pool_and_fill_packed_batches(dataset_size):
    mix = get_merged_adapter_with_data_mix(EntropyGainProportionalSampler, dataset_size=dataset_size)
    reasoning_size, direct_size = [adapter.sampled_size() for adapter in mix.dataset_adapters]

    assert reasoning_size == dataset_size // 10
    assert abs(direct_size - reasoning_size // 4) <= 32
    assert mix.sampled_size() % 64 == 0


def test_math_mix_uses_fresh_draws_and_preserves_distilled_and_ground_truth_targets(thinking_tokenizer, tmp_path):
    rng = Random(7)
    source = pd.DataFrame(
        [
            {
                "question_id": f"trace_{i}",
                "question": "Simplify the expression in terms of A and B.",
                "answer": r"\frac{A}{B}",
                "distill_reasoning": "Cancel the common factors.",
                "distill_answer": r"\frac{B}{A}",
                "entropy_value": 0.8 if i < 320 else 0.1,
                "teacher_entropy": 0.2,
                "random_value": rng.random(),
            }
            for i in range(2560)
        ]
    )
    mix = get_merged_adapter_with_data_mix(
        EntropyGainProportionalSampler,
        dataset_size=len(source),
        dataset_cls=HendrycksMathReasoningResponseDatasetDetailedPrompt,
    )
    mix.override_tokenizer(thinking_tokenizer)
    reasoning_size, direct_size = [adapter.sampled_size() for adapter in mix.dataset_adapters]
    eligible_ids = set(source.iloc[:320]["question_id"])
    reasoning_selections = []

    for epoch in range(2):
        source["random_value"] = [rng.random() for _ in range(len(source))]
        path = tmp_path / f"entropy_epoch_{epoch}.parquet"
        source.to_parquet(path, index=False)
        before = path.read_bytes()
        rows = mix.process_dataset(path_override=str(path), shuffle=True, shuffle_seed=42 + epoch)
        reasoning_rows = rows.select(range(reasoning_size))
        direct_rows = rows.select(range(reasoning_size, len(rows)))
        reasoning_ids = set(reasoning_rows["row_id"])
        direct_ids = set(direct_rows["row_id"])

        assert len(rows) == mix.sampled_size()
        assert len(direct_rows) == direct_size
        assert direct_ids < reasoning_ids <= eligible_ids
        reasoning_selections.append(reasoning_ids)

        for adapter, selected_rows, expected_target in (
            (mix.dataset_adapters[0], reasoning_rows, r"<think>Cancel the common factors.</think>\frac{B}{A}"),
            (mix.dataset_adapters[1], direct_rows, r"\frac{A}{B}"),
        ):
            prompt = QADatasetAdapter(adapter.dataset).process_row(source.iloc[0].to_dict()).input_ids
            for row in selected_rows:
                assert row["input_ids"][:len(prompt)] == prompt
                assert row["labels"][:len(prompt)] == [-100] * len(prompt)
                target = thinking_tokenizer.decode(row["labels"][len(prompt):])
                assert target.startswith(expected_target)
        assert path.read_bytes() == before

    assert reasoning_selections[0] != reasoning_selections[1]
