# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SampleStateArrays:
    """Dense in-memory state used by the online sampler."""

    sample_ids: np.ndarray
    latest_sigma: np.ndarray
    ema_sigma: np.ndarray
    last_observed_step: np.ndarray
    observation_count: np.ndarray

    @classmethod
    def empty(cls, sample_ids: Iterable[str]) -> SampleStateArrays:
        sample_ids_array = np.asarray(list(sample_ids), dtype=object)
        size = len(sample_ids_array)
        return cls(
            sample_ids=sample_ids_array,
            latest_sigma=np.zeros(size, dtype=np.float64),
            ema_sigma=np.zeros(size, dtype=np.float64),
            last_observed_step=np.zeros(size, dtype=np.int64),
            observation_count=np.zeros(size, dtype=np.int64),
        )

    def __post_init__(self):
        expected = len(self.sample_ids)
        for name in ("latest_sigma", "ema_sigma", "last_observed_step", "observation_count"):
            if len(getattr(self, name)) != expected:
                raise ValueError(f"{name} length must match sample_ids")
        self._sample_id_to_index = {str(sample_id): index for index, sample_id in enumerate(self.sample_ids)}
        if len(self._sample_id_to_index) != expected:
            raise ValueError("sample_ids must be unique")

    def update(self, sample_id: str, sigma: float, policy_step: int, ema_alpha: float) -> None:
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        index = self._sample_id_to_index[sample_id]
        previous_count = int(self.observation_count[index])
        previous_ema = float(self.ema_sigma[index])
        self.latest_sigma[index] = sigma
        self.ema_sigma[index] = sigma if previous_count == 0 else ema_alpha * sigma + (1 - ema_alpha) * previous_ema
        self.last_observed_step[index] = policy_step
        self.observation_count[index] = previous_count + 1

    def index_for(self, sample_id: str) -> int:
        return self._sample_id_to_index[sample_id]


def compute_sampling_weights(
    states: SampleStateArrays,
    *,
    current_step: int,
    sigma_scale: float,
    min_weight: float,
    staleness_weight: float,
    staleness_horizon: int,
) -> np.ndarray:
    """Compute positive latest-dominant reward-variance and staleness weights."""

    if current_step < 0:
        raise ValueError("current_step must be non-negative")
    if min_weight <= 0:
        raise ValueError("min_weight must be positive")
    if staleness_weight < 0:
        raise ValueError("staleness_weight must be non-negative")
    if staleness_horizon <= 0:
        raise ValueError("staleness_horizon must be positive")

    scale = max(float(sigma_scale), 1e-12)
    sigma_component = np.clip(states.ema_sigma / scale, 0.0, 1.0)
    age = np.maximum(0, current_step - states.last_observed_step)
    staleness_component = np.minimum(age / staleness_horizon, 1.0)
    weights = min_weight + sigma_component + staleness_weight * staleness_component
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("sampling weights must be finite and positive")
    return weights


def weighted_sample_without_replacement(
    weights: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    exclude_indices: Iterable[int] = (),
) -> np.ndarray:
    """Sample dense indices without replacement while respecting exclusions."""

    if count < 0:
        raise ValueError("count must be non-negative")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")

    eligible = np.ones(len(weights), dtype=bool)
    excluded = np.asarray(list(exclude_indices), dtype=np.int64)
    if excluded.size:
        if np.any(excluded < 0) or np.any(excluded >= len(weights)):
            raise IndexError("exclude_indices contains an out-of-range index")
        eligible[excluded] = False

    candidates = np.flatnonzero(eligible)
    if count == 0 or not candidates.size:
        return np.empty(0, dtype=np.int64)
    count = min(count, len(candidates))
    candidate_weights = weights[candidates]
    if not np.all(np.isfinite(candidate_weights)) or np.any(candidate_weights <= 0):
        raise ValueError("eligible weights must be finite and positive")
    probabilities = candidate_weights / candidate_weights.sum()
    return rng.choice(candidates, size=count, replace=False, p=probabilities).astype(np.int64, copy=False)
