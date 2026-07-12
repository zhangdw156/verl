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

import math
from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True)
class RolloutObservation:
    """Raw outcome for one rollout in a prompt group."""

    rollout_index: int
    reward: float
    correct: int
    format: int
    status: str = "success"
    response: str | None = None

    def __post_init__(self):
        if self.rollout_index < 0:
            raise ValueError("rollout_index must be non-negative")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if self.correct not in (0, 1):
            raise ValueError("correct must be 0 or 1")
        if self.format not in (0, 1):
            raise ValueError("format must be 0 or 1")


@dataclass(frozen=True)
class SampleObservation:
    """A complete n-rollout observation for one persistent sample."""

    sample_id: str
    dense_index: int
    phase: str
    policy_step: int
    generation_round: int
    rollouts: tuple[RolloutObservation, ...]
    attempt_id: str = "default"

    def __post_init__(self):
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.dense_index < 0:
            raise ValueError("dense_index must be non-negative")
        if self.phase not in {"step0", "train"}:
            raise ValueError("phase must be 'step0' or 'train'")
        if self.policy_step < 0:
            raise ValueError("policy_step must be non-negative")
        if self.generation_round < 0:
            raise ValueError("generation_round must be non-negative")
        if not self.attempt_id:
            raise ValueError("attempt_id must not be empty")
        if not self.rollouts:
            raise ValueError("rollouts must not be empty")
        rollout_indices = [item.rollout_index for item in self.rollouts]
        if len(rollout_indices) != len(set(rollout_indices)):
            raise ValueError("rollout_index values must be unique within a sample observation")

    @property
    def n(self) -> int:
        return len(self.rollouts)

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(item.reward for item in self.rollouts)

    @property
    def corrects(self) -> tuple[int, ...]:
        return tuple(item.correct for item in self.rollouts)

    @property
    def formats(self) -> tuple[int, ...]:
        return tuple(item.format for item in self.rollouts)

    @property
    def p(self) -> float:
        return fmean(self.corrects)

    @property
    def q(self) -> float:
        return fmean(self.formats)

    @property
    def mu(self) -> float:
        return fmean(self.rewards)

    @property
    def sigma(self) -> float:
        return pstdev(self.rewards)

    def is_zero_variance(self, epsilon: float) -> bool:
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        return self.sigma <= epsilon
