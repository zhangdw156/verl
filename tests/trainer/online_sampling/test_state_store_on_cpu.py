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

import sqlite3

import pytest

from verl.trainer.online_sampling.state_store import SQLiteOnlineStateStore
from verl.trainer.online_sampling.types import RolloutObservation, SampleObservation


def make_observation(
    sample_id: str,
    dense_index: int,
    rewards: tuple[float, ...],
    *,
    policy_step: int = 0,
    phase: str = "step0",
    generation_round: int = 0,
) -> SampleObservation:
    return SampleObservation(
        sample_id=sample_id,
        dense_index=dense_index,
        phase=phase,
        policy_step=policy_step,
        generation_round=generation_round,
        rollouts=tuple(
            RolloutObservation(
                rollout_index=index,
                reward=reward,
                correct=int(reward >= 1.0),
                format=int(reward > 0),
            )
            for index, reward in enumerate(rewards)
        ),
    )


def test_store_appends_raw_and_aggregate_history(tmp_path):
    store = SQLiteOnlineStateStore(tmp_path / "state.sqlite")
    store.ensure_run("run", {"model": "m"})
    store.register_samples("run", [("sample", 0, {"source_row": 1}, True)])

    state = store.append_observation(
        "run",
        make_observation("sample", 0, (0.0, 1.1)),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )

    assert state["latest_sigma"] == pytest.approx(0.55)
    assert state["ema_sigma"] == pytest.approx(0.55)
    assert store.observation_count("run", "step0") == 1
    assert store.pending_step0_sample_ids("run") == []
    integrity, foreign_keys = store.integrity_check()
    assert integrity == "ok"
    assert foreign_keys == []

    connection = sqlite3.connect(store.path)
    assert connection.execute("SELECT COUNT(*) FROM rollout_observations").fetchone()[0] == 2
    assert connection.execute("SELECT p, q, mu, sigma FROM sample_observations").fetchone() == pytest.approx(
        (0.5, 0.5, 0.55, 0.55)
    )
    connection.close()
    store.close()


def test_store_preserves_history_and_updates_ema(tmp_path):
    store = SQLiteOnlineStateStore(tmp_path / "state.sqlite")
    store.ensure_run("run", {"model": "m"})
    store.register_samples("run", [("sample", 0, None, True)])

    store.append_observation(
        "run",
        make_observation("sample", 0, (0.0, 1.0)),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )
    state = store.append_observation(
        "run",
        make_observation(
            "sample",
            0,
            (1.0, 1.0),
            policy_step=1,
            phase="train",
        ),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )

    assert store.observation_count("run") == 2
    assert state["latest_sigma"] == 0.0
    assert state["ema_sigma"] == pytest.approx(0.1)
    assert state["observation_count"] == 2
    store.close()


def test_store_is_idempotent_for_same_observation_group(tmp_path):
    store = SQLiteOnlineStateStore(tmp_path / "state.sqlite")
    store.ensure_run("run", {"model": "m"})
    store.register_samples("run", [("sample", 0, None, True)])
    observation = make_observation("sample", 0, (0.0, 1.0))

    store.append_observation(
        "run",
        observation,
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )
    store.append_observation(
        "run",
        observation,
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )

    assert store.observation_count("run") == 1
    assert store.get_latest_state("run", "sample")["observation_count"] == 1
    store.close()


def test_store_rejects_fingerprint_mismatch(tmp_path):
    store = SQLiteOnlineStateStore(tmp_path / "state.sqlite")
    store.ensure_run("run", {"model": "a"})

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        store.ensure_run("run", {"model": "b"})

    store.close()


def test_checkpoint_snapshot_restores_checkpoint_aligned_state(tmp_path):
    store = SQLiteOnlineStateStore(tmp_path / "state.sqlite")
    store.ensure_run("run", {"model": "m"})
    store.register_samples("run", [("sample", 0, None, True)])

    store.append_observation(
        "run",
        make_observation("sample", 0, (0.0, 1.0)),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )
    store.save_checkpoint_snapshot("run", "latest", 0)
    store.append_observation(
        "run",
        make_observation("sample", 0, (1.0, 1.0), phase="train", policy_step=1),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )

    store.restore_checkpoint_snapshot("run", "latest", 0)
    restored = store.load_latest_states("run")

    assert restored[0]["latest_sigma"] == pytest.approx(0.5)
    assert restored[0]["ema_sigma"] == pytest.approx(0.5)
    assert restored[0]["observation_count"] == 1

    # Replaying the already persisted post-checkpoint group reactivates it exactly once.
    replayed = store.append_observation(
        "run",
        make_observation("sample", 0, (1.0, 1.0), phase="train", policy_step=1),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )
    assert replayed["latest_sigma"] == 0.0
    assert replayed["ema_sigma"] == pytest.approx(0.1)
    assert replayed["observation_count"] == 2
    assert store.observation_count("run") == 2

    fresh_attempt = make_observation(
        "sample",
        0,
        (0.0, 1.0),
        phase="train",
        policy_step=1,
    )
    fresh_attempt = SampleObservation(
        sample_id=fresh_attempt.sample_id,
        dense_index=fresh_attempt.dense_index,
        phase=fresh_attempt.phase,
        policy_step=fresh_attempt.policy_step,
        generation_round=fresh_attempt.generation_round,
        rollouts=fresh_attempt.rollouts,
        attempt_id="new-process",
    )
    store.append_observation(
        "run",
        fresh_attempt,
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )
    assert store.observation_count("run") == 3
    store.close()


def test_checkpoint_snapshot_pruning_keeps_newest_steps(tmp_path):
    store = SQLiteOnlineStateStore(tmp_path / "state.sqlite")
    store.ensure_run("run", {"model": "m"})
    store.register_samples("run", [("sample", 0, None, True)])
    store.append_observation(
        "run",
        make_observation("sample", 0, (0.0, 1.0)),
        ema_alpha=0.8,
        zero_variance_epsilon=1e-6,
        save_responses=False,
    )
    for step in (1, 2, 3):
        store.save_checkpoint_snapshot("run", "latest", step)

    store.prune_checkpoint_snapshots("run", "latest", keep=2)

    assert not store.checkpoint_snapshot_exists("run", "latest", 1)
    assert store.checkpoint_snapshot_exists("run", "latest", 2)
    assert store.checkpoint_snapshot_exists("run", "latest", 3)
    store.close()
