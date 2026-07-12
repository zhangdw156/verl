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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verl.trainer.config.algorithm import OnlineSamplingConfig

from .priority import SampleStateArrays, compute_sampling_weights, weighted_sample_without_replacement
from .state_store import SQLiteOnlineStateStore
from .types import SampleObservation


class OnlineSamplingController:
    """Own persistent sample state and deterministic online sampling decisions."""

    def __init__(
        self,
        *,
        config: OnlineSamplingConfig,
        run_id: str,
        fingerprint: Mapping[str, Any],
        sample_ids: Sequence[str],
        sample_metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ):
        if not config.enabled:
            raise ValueError("OnlineSamplingController requires online_sampling.enabled=true")
        if not config.state_path:
            raise ValueError("online_sampling.state_path must be configured")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample_ids must be unique")
        if sample_metadata is not None and len(sample_metadata) != len(sample_ids):
            raise ValueError("sample_metadata length must match sample_ids")

        self.config = config
        self.run_id = run_id
        self.store = SQLiteOnlineStateStore(Path(config.state_path))
        self.store.ensure_run(run_id, fingerprint)
        metadata = sample_metadata or [None] * len(sample_ids)
        self.store.register_samples(
            run_id,
            ((sample_id, dense_index, metadata[dense_index], True) for dense_index, sample_id in enumerate(sample_ids)),
        )
        self.states = self._load_states(self.store.load_latest_states(run_id))
        self.sigma_scale = float(self.store.get_run_value(run_id, "sigma_scale", 1.0))

    def _load_states(self, rows: list[dict[str, Any]]) -> SampleStateArrays:
        states = SampleStateArrays.empty(row["sample_id"] for row in rows)
        states.latest_sigma[:] = [row["latest_sigma"] for row in rows]
        states.ema_sigma[:] = [row["ema_sigma"] for row in rows]
        states.last_observed_step[:] = [row["last_observed_step"] for row in rows]
        states.observation_count[:] = [row["observation_count"] for row in rows]
        return states

    def pending_step0_indices(self) -> np.ndarray:
        sample_ids = self.store.pending_step0_sample_ids(self.run_id)
        return np.asarray([self.states.index_for(sample_id) for sample_id in sample_ids], dtype=np.int64)

    def record_observation(self, observation: SampleObservation) -> bool:
        if observation.n != self.config.rollout_n:
            raise ValueError(
                f"Expected {self.config.rollout_n} rollouts for sample {observation.sample_id}, got {observation.n}"
            )
        state = self.store.append_observation(
            self.run_id,
            observation,
            ema_alpha=self.config.ema_alpha,
            zero_variance_epsilon=self.config.zero_variance_epsilon,
            save_responses=self.config.save_responses,
        )
        index = self.states.index_for(observation.sample_id)
        self.states.latest_sigma[index] = state["latest_sigma"]
        self.states.ema_sigma[index] = state["ema_sigma"]
        self.states.last_observed_step[index] = state["last_observed_step"]
        self.states.observation_count[index] = state["observation_count"]
        return not observation.is_zero_variance(self.config.zero_variance_epsilon)

    def finish_step0(self) -> float:
        pending = self.pending_step0_indices()
        if pending.size:
            raise RuntimeError(f"Cannot finish Step 0 with {len(pending)} pending samples")
        positive = self.states.latest_sigma[self.states.latest_sigma > 0]
        self.sigma_scale = float(np.percentile(positive, 95)) if positive.size else 1.0
        self.store.set_run_value(self.run_id, "sigma_scale", self.sigma_scale)
        self.store.set_run_value(self.run_id, "step0_complete", True)
        self.store.save_checkpoint_snapshot(self.run_id, "step0", 0)
        return self.sigma_scale

    def is_step0_complete(self) -> bool:
        marked_complete = bool(self.store.get_run_value(self.run_id, "step0_complete", False))
        return marked_complete and not self.pending_step0_indices().size

    def sampling_weights(self, current_step: int) -> np.ndarray:
        return compute_sampling_weights(
            self.states,
            current_step=current_step,
            sigma_scale=self.sigma_scale,
            min_weight=self.config.min_weight,
            staleness_weight=self.config.staleness_weight,
            staleness_horizon=self.config.staleness_horizon,
        )

    def sample_indices(
        self,
        *,
        current_step: int,
        generation_round: int,
        count: int,
        exclude_indices: Sequence[int] = (),
    ) -> np.ndarray:
        seed = np.random.SeedSequence([self.config.seed, current_step, generation_round])
        rng = np.random.default_rng(seed)
        return weighted_sample_without_replacement(
            self.sampling_weights(current_step),
            count=count,
            rng=rng,
            exclude_indices=exclude_indices,
        )

    def save_checkpoint_snapshot(self, checkpoint_kind: str, checkpoint_step: int) -> None:
        self.store.save_checkpoint_snapshot(self.run_id, checkpoint_kind, checkpoint_step)

    def restore_checkpoint_snapshot(self, checkpoint_kind: str, checkpoint_step: int) -> None:
        self.store.restore_checkpoint_snapshot(self.run_id, checkpoint_kind, checkpoint_step)
        rows = self.store.load_checkpoint_snapshot(self.run_id, checkpoint_kind, checkpoint_step)
        self.states = self._load_states(rows)

    def delete_checkpoint_snapshot(self, checkpoint_kind: str, checkpoint_step: int) -> None:
        self.store.delete_checkpoint_snapshot(self.run_id, checkpoint_kind, checkpoint_step)

    def prune_checkpoint_snapshots(self, checkpoint_kind: str, keep: int) -> None:
        self.store.prune_checkpoint_snapshots(self.run_id, checkpoint_kind, keep)

    def close(self) -> None:
        self.store.close()
