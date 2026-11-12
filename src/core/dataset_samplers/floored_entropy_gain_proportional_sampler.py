"""Proportional sampling from ``floor + max(student - proxy, 0)``; measured rows are never dropped."""

from math import isfinite, log
from random import random
from typing import override

import pandas as pd
from pydantic import model_validator

from core.dataset_samplers.base_sampler import BaseDatasetSampler, BaseDatasetSamplerConfig
from core.dataset_samplers.entropy_gain_proportional_sampler import U_FLOOR


class FlooredEntropyGainProportionalSamplerConfig(BaseDatasetSamplerConfig):
    # Share of the total draw weight that is uniform across rows. Each epoch the floor is
    # uniform_share / (1 - uniform_share) * mean(max(gain, 0)), so floor / (floor + mean) = share.
    uniform_share: float = 0.1

    @model_validator(mode="after")
    def _validate_share(self):
        if not isfinite(self.uniform_share) or not 0 < self.uniform_share < 1:
            raise ValueError("uniform_share must be in (0, 1)")
        return self


def _gain(row) -> float:
    try:
        gain = float(row["entropy_value"]) - float(row["teacher_entropy"])
    except (TypeError, ValueError):
        return float("nan")
    return gain if isfinite(gain) else float("nan")


class FlooredEntropyGainProportionalSampler(BaseDatasetSampler):
    """EntropyGainProportionalSampler with weight ``floor + max(gain, 0)`` instead of ``max(gain, 0)``.

    The floor is recomputed from the pool every epoch so that a fixed ``uniform_share`` of the
    draw weight is uniform, however large or small the gaps are. Same Efraimidis-Spirakis draw
    (weight / Exp(1), ``random_value`` from SingleTokenEntropyWithRandomEstimator). Rows the
    student has caught up on stay drawable, so each epoch fills top_k; unmeasured rows still
    score NaN and are dropped.
    """

    config: FlooredEntropyGainProportionalSamplerConfig

    def __init__(self, config: BaseDatasetSamplerConfig):
        if not isinstance(config, FlooredEntropyGainProportionalSamplerConfig):
            config = FlooredEntropyGainProportionalSamplerConfig(**config.model_dump())
        super().__init__(config)
        self.floor = 1.0

    def floor_for(self, df: pd.DataFrame) -> float:
        mean_gain = df.apply(_gain, axis=1).clip(lower=0).mean()  # NaN rows are skipped
        if not mean_gain > 0:  # every gap <= 0 (or none measured): sample uniformly
            return 1.0
        share = self.config.uniform_share
        return share / (1 - share) * float(mean_gain)

    @override
    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        self.floor = self.floor_for(df)
        return super()._select(df)

    @override
    def _score_row(self, row: dict) -> float:
        gain = _gain(row)
        if not isfinite(gain):
            return float("nan")

        u = float(row["random_value"])
        if not 0.0 < u < 1.0:
            # NaN draw from a failed measurement; redraw as EntropyGainProportionalSampler does.
            u = random()
        return (self.floor + max(gain, 0.0)) / -log(max(u, U_FLOOR))
