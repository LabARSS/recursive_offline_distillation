"""GSM8K direct-answer redraws through the experiment's actual mixed adapter."""

import pandas as pd
import pytest

from core.dataset_samplers.entropy_gain_proportional_sampler import EntropyGainProportionalSampler
from core.datasets.qa_dataset_adapter import QADatasetAdapter
from core.training.resampling_trainer import ResamplingDataset
from experiments.distillation_by_metrics.gsm8k.shared import get_merged_adapter_with_data_mix


@pytest.mark.parametrize("redraw_each_epoch", [False, True])
def test_gsm8k_mix_redraws_and_supervises_direct_answers(
    thinking_tokenizer, tmp_path, redraw_each_epoch
):
    source = pd.DataFrame(
        [
            {
                "question_id": str(i),
                "question": f"What is {i} plus one?",
                "answer": str(i + 1),
                "distill_reasoning": f"Adding one to {i} gives {i + 1}.",
                "distill_answer": str(i + 1),
                "entropy_value": 0.8,
                "teacher_entropy": 0.2,
                "random_value": (i + 1) / 1201,
            }
            for i in range(1200)
        ]
    )
    path = tmp_path / "cached_entropy.parquet"
    source.to_parquet(path, index=False)
    before = path.read_bytes()

    mix = get_merged_adapter_with_data_mix(EntropyGainProportionalSampler)
    mix.override_tokenizer(thinking_tokenizer)
    dataset = ResamplingDataset(
        mix, thinking_tokenizer, shuffle=True, data_seed=42,
        redraw_each_epoch=redraw_each_epoch,
    )
    dataset.dataset_path = str(path)
    dataset.selection_epoch = 0  # Both training epochs reuse the same entropy estimate.
    dataset.epoch = 7
    first = list(dataset)
    replay = list(dataset)
    dataset.epoch = 8
    later = list(dataset)

    reasoning_size, direct_size = [adapter.sampled_size() for adapter in mix.dataset_adapters]
    assert (reasoning_size, direct_size) == (768, 256)
    assert len(first) == len(later) == len(dataset) == 1024
    assert first == replay
    first_direct = {row["row_id"] for row in first[reasoning_size:]}
    later_direct = {row["row_id"] for row in later[reasoning_size:]}
    assert (first_direct != later_direct) == redraw_each_epoch

    probe = QADatasetAdapter(mix.dataset_adapters[1].dataset)
    by_id = source.set_index("question_id", drop=False)
    for rows in (first, later):
        reasoning_ids = {row["row_id"] for row in rows[:reasoning_size]}
        direct_ids = {row["row_id"] for row in rows[reasoning_size:]}
        assert len(direct_ids) == direct_size
        assert direct_ids < reasoning_ids
        for row in rows[reasoning_size:]:
            original = by_id.loc[row["row_id"]].to_dict()
            prompt = probe.process_row(original).input_ids
            assert row["input_ids"][:len(prompt)] == prompt
            assert row["labels"][:len(prompt)] == [-100] * len(prompt)
            answer_tokens = row["labels"][len(prompt):]
            assert answer_tokens == row["input_ids"][len(prompt):]
            assert thinking_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip() == original["answer"]
            assert thinking_tokenizer.thinking_start_token_id not in answer_tokens

    assert path.read_bytes() == before


def test_gsm8k_direct_mix_size_is_capped_by_positive_gain_candidates():
    mix = get_merged_adapter_with_data_mix(EntropyGainProportionalSampler)
    source = pd.DataFrame(
        {
            "entropy_value": [0.8] * 100 + [0.1] * 400 + [float("nan")] * 100,
            "teacher_entropy": [0.2] * 600,
            "random_value": [0.5] * 600,
        }
    )

    assert mix.selected_sample_counts(source) == {
        "train_gsm8k_reasoning": 100,
        "train_gsm8k_direct_response": 100,
    }
