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

from verl.trainer.config.algorithm import OnlineSamplingConfig
from verl.trainer.online_sampling.controller import OnlineSamplingController
from verl.trainer.online_sampling.types import RolloutObservation, SampleObservation


def make_observation(sample_id: str, dense_index: int, rewards: tuple[float, ...]) -> SampleObservation:
    return SampleObservation(
        sample_id=sample_id,
        dense_index=dense_index,
        phase="step0",
        policy_step=0,
        generation_round=dense_index,
        rollouts=tuple(
            RolloutObservation(
                rollout_index=index,
                reward=reward,
                correct=int(reward >= 1),
                format=int(reward > 0),
            )
            for index, reward in enumerate(rewards)
        ),
    )


def test_controller_resumes_step0_and_samples_deterministically(tmp_path):
    config = OnlineSamplingConfig(
        enabled=True,
        state_path=str(tmp_path / "state.sqlite"),
        rollout_n=2,
        seed=7,
    )
    controller = OnlineSamplingController(
        config=config,
        run_id="run",
        fingerprint={"model": "m"},
        sample_ids=["a", "b", "c"],
    )

    assert controller.pending_step0_indices().tolist() == [0, 1, 2]
    assert controller.record_observation(make_observation("a", 0, (0.0, 1.0)))
    assert not controller.record_observation(make_observation("b", 1, (1.0, 1.0)))
    assert controller.record_observation(make_observation("c", 2, (0.0, 1.0)))
    assert controller.pending_step0_indices().size == 0
    assert controller.finish_step0() == pytest.approx(0.5)
    first = controller.sample_indices(current_step=1, generation_round=0, count=2)
    controller.close()

    resumed = OnlineSamplingController(
        config=config,
        run_id="run",
        fingerprint={"model": "m"},
        sample_ids=["a", "b", "c"],
    )
    second = resumed.sample_indices(current_step=1, generation_round=0, count=2)

    assert resumed.is_step0_complete()
    assert np.array_equal(first, second)
    resumed.close()
