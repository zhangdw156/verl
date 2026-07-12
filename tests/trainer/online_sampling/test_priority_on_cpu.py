# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from verl.trainer.online_sampling.priority import (
    SampleStateArrays,
    compute_sampling_weights,
    weighted_sample_without_replacement,
)


def test_latest_dominant_ema_update():
    states = SampleStateArrays.empty(["a"])
    states.update("a", sigma=0.5, policy_step=0, ema_alpha=0.8)
    states.update("a", sigma=0.0, policy_step=1, ema_alpha=0.8)

    assert states.latest_sigma[0] == 0.0
    assert states.ema_sigma[0] == pytest.approx(0.1)
    assert states.last_observed_step[0] == 1
    assert states.observation_count[0] == 2


def test_weights_include_positive_floor_and_capped_staleness():
    states = SampleStateArrays.empty(["a", "b"])
    states.ema_sigma[:] = [0.5, 0.0]
    states.last_observed_step[:] = [10, 0]

    weights = compute_sampling_weights(
        states,
        current_step=100,
        sigma_scale=0.5,
        min_weight=0.001,
        staleness_weight=0.1,
        staleness_horizon=50,
    )

    assert weights[0] == pytest.approx(1.101)
    assert weights[1] == pytest.approx(0.101)
    assert np.all(weights > 0)


def test_weighted_sampling_is_deterministic_and_without_replacement():
    weights = np.array([1.0, 2.0, 3.0, 4.0])

    first = weighted_sample_without_replacement(
        weights,
        count=3,
        rng=np.random.default_rng(123),
        exclude_indices=[1],
    )
    second = weighted_sample_without_replacement(
        weights,
        count=3,
        rng=np.random.default_rng(123),
        exclude_indices=[1],
    )

    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 3
    assert 1 not in first


def test_weighted_sampling_caps_count_to_available_items():
    sampled = weighted_sample_without_replacement(
        np.ones(3),
        count=10,
        rng=np.random.default_rng(0),
        exclude_indices=[0],
    )

    assert sorted(sampled.tolist()) == [1, 2]
