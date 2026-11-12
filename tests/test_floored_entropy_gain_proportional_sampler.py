"""Floored entropy-gain proportional sampling: weight floor + max(gain, 0), floor set by uniform share."""

import random
from math import isnan, log, nan

import pandas as pd
import pytest

from core.dataset_samplers.base_sampler import BaseDatasetSamplerConfig
from core.dataset_samplers.floored_entropy_gain_proportional_sampler import (
    FlooredEntropyGainProportionalSampler,
    FlooredEntropyGainProportionalSamplerConfig,
)


def _row(student, proxy, u=0.5):
    return {"entropy_value": student, "teacher_entropy": proxy, "random_value": u}


def test_floor_gives_the_configured_uniform_share():
    sampler = FlooredEntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))
    df = pd.DataFrame([_row(0.4, 0.1), _row(0.2, 0.1), _row(0.0, 0.5), _row(nan, 0.1)])
    floor = sampler.floor_for(df)
    mean_gain = (0.3 + 0.1 + 0.0) / 3  # NaN row skipped, negative gap clamped to 0
    assert floor == pytest.approx(mean_gain / 9)
    assert floor / (floor + mean_gain) == pytest.approx(0.1)


def test_score_is_floored_gain_over_an_exponential_variate():
    sampler = FlooredEntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=1))
    sampler.floor = 0.01
    assert sampler._score_row(_row(0.9, 0.2)) == pytest.approx(0.71 / -log(0.5))
    assert sampler._score_row(_row(0.1, 0.1)) == pytest.approx(0.01 / -log(0.5))
    assert sampler._score_row(_row(0.0, 0.9)) == pytest.approx(0.01 / -log(0.5))


def test_all_non_positive_gaps_fall_back_to_uniform():
    sampler = FlooredEntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=2))
    df = pd.DataFrame([_row(0.0, 0.5), _row(0.1, 0.1), _row(0.2, 0.3)])
    assert sampler.floor_for(df) == 1.0
    assert sampler.count_selected(df) == 2


def test_base_config_is_upgraded_with_default_share():
    sampler = FlooredEntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=3))
    assert sampler.config.top_k == 3 and sampler.config.uniform_share == 0.1


def test_negative_gain_rows_still_fill_top_k():
    sampler = FlooredEntropyGainProportionalSampler(FlooredEntropyGainProportionalSamplerConfig(top_k=4))
    df = pd.DataFrame([_row(0.5, 0.1), _row(0.05, 0.5), _row(0.0, 0.4), _row(0.1, 0.1), _row(0.2, 0.9)])
    df["random_value"] = [random.random() for _ in range(len(df))]
    assert sampler.count_selected(df) == 4


def test_unmeasured_rows_are_dropped():
    sampler = FlooredEntropyGainProportionalSampler(BaseDatasetSamplerConfig(top_k=2))
    assert isnan(sampler._score_row(_row(nan, 0.1)))
    assert isnan(sampler._score_row(_row(None, 0.1)))
    assert sampler.count_selected(pd.DataFrame([_row(nan, 0.1), _row(0.0, 0.5), _row(0.5, 0.1)])) == 2


def test_draw_frequency_matches_weights():
    sampler = FlooredEntropyGainProportionalSampler(FlooredEntropyGainProportionalSamplerConfig(top_k=1))
    rows = [_row(0.18, 0.0), _row(0.0, 0.3)]  # mean gain 0.09 -> floor 0.01; weights 0.19 vs 0.01
    random.seed(0)
    wins = 0
    for _ in range(20000):
        df = pd.DataFrame(rows)
        df["random_value"] = [random.random() for _ in rows]
        wins += sampler._select(df).index[0] == 1
    assert sampler.floor == pytest.approx(0.01)
    assert wins / 20000 == pytest.approx(0.01 / 0.20, abs=0.005)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, nan])
def test_invalid_share_is_rejected(value):
    with pytest.raises(ValueError):
        FlooredEntropyGainProportionalSamplerConfig(top_k=1, uniform_share=value)
