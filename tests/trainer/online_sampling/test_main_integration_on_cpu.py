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

import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import verl.trainer.main_ppo_sync as main_ppo_sync
from verl.trainer.config.algorithm import OnlineSamplingConfig
from verl.trainer.main_ppo_sync import (
    PPOTrainer,
    ReplayBuffer,
    _model_path_fingerprint,
    _online_sampling_state_semantics,
)


class FakeBatch:
    partition_id = "train"
    keys = [
        "uid-a_0_0",
        "uid-a_1_0",
        "uid-b_0_0",
        "uid-b_1_0",
    ]
    tags = [{"status": "success"} for _ in keys]


class FakeTQ:
    @staticmethod
    def kv_batch_get(keys, partition_id, select_fields):
        del keys, partition_id, select_fields
        return {
            "uid": ["uid-a", "uid-a", "uid-b", "uid-b"],
            "sample_id": ["sample-a", "sample-a", "sample-b", "sample-b"],
            "index": [0, 0, 1, 1],
            "rm_scores": torch.tensor([[0.0], [1.0], [1.0], [1.0]]),
            "extra_fields": [
                {"reward_extra_info": {"correct": 0, "format": 0}},
                {"reward_extra_info": {"correct": 1, "format": 1}},
                {"reward_extra_info": {"correct": 1, "format": 1}},
                {"reward_extra_info": {"correct": 1, "format": 1}},
            ],
        }


def test_replay_buffer_sample_isolates_expected_prompt_uids():
    replay_buffer = object.__new__(ReplayBuffer)
    replay_buffer.partitions = defaultdict(dict)
    replay_buffer.retired_uids = defaultdict(set)
    replay_buffer.lock = threading.Lock()
    replay_buffer.poll_interval = 0
    replay_buffer.retire("train", {"old"})
    replay_buffer.add(
        "train",
        {
            "old_0_0": {"global_steps": 0, "status": "success"},
            "old": {"status": "finished"},
            "new_0_0": {"global_steps": 0, "status": "success"},
            "new": {"global_steps": 0, "status": "finished"},
            "other_0_0": {"global_steps": 0, "status": "success"},
            "other": {"global_steps": 0, "status": "finished"},
        },
    )

    batch = replay_buffer.sample(
        partition_id="train",
        global_steps=0,
        expected_uids={"new"},
    )

    assert batch.keys == ["new_0_0"]
    assert set(replay_buffer.partitions["train"]) == {
        "new_0_0",
        "new",
        "other_0_0",
        "other",
    }


def test_replay_buffer_reports_failed_expected_prompt():
    replay_buffer = object.__new__(ReplayBuffer)
    replay_buffer.partitions = defaultdict(dict)
    replay_buffer.retired_uids = defaultdict(set)
    replay_buffer.lock = threading.Lock()
    replay_buffer.poll_interval = 0
    replay_buffer.add(
        "train",
        {
            "failed_0_0": {"global_steps": 0, "status": "success"},
            "failed": {"global_steps": 0, "status": "failure"},
        },
    )

    batch = replay_buffer.sample(
        partition_id="train",
        global_steps=0,
        expected_uids={"failed"},
    )

    assert batch.keys == []
    assert batch.extra_info["failed_uids"] == ["failed"]
    assert batch.extra_info["failed_output_keys"] == ["failed_0_0"]


def test_sample_replay_batch_raises_for_training_prompt_failure(monkeypatch):
    class FailedReplayBuffer:
        def __init__(self):
            self.retired = []

        def sample(self, partition_id, global_steps, expected_uids):
            del global_steps, expected_uids
            return main_ppo_sync.KVBatchMeta(
                partition_id=partition_id,
                keys=["good_0_0"],
                tags=[{"global_steps": 0, "status": "success"}],
                extra_info={
                    "failed_uids": ["failed"],
                    "failed_output_keys": ["failed_0_0"],
                },
            )

        def retire(self, partition_id, prompt_uids):
            self.retired.append((partition_id, prompt_uids))

    class ClearTQ:
        cleared = []

        @classmethod
        def kv_clear(cls, keys, partition_id):
            cls.cleared.append((partition_id, list(keys)))

    trainer = object.__new__(PPOTrainer)
    trainer.global_steps = 0
    trainer.replay_buffer = FailedReplayBuffer()
    monkeypatch.setattr(main_ppo_sync, "tq", ClearTQ)

    with pytest.raises(RuntimeError, match="Generation failed for 1 prompts"):
        trainer._sample_replay_batch(
            partition_id="train",
            prompt_uids=["good", "failed"],
        )

    assert ClearTQ.cleared == [
        ("train", ["good", "failed", "failed_0_0"]),
        ("train", ["good_0_0"]),
    ]
    assert trainer.replay_buffer.retired == [("train", ["good", "failed"])]


def test_sample_replay_batch_allows_validation_prompt_failure(monkeypatch):
    class FailedReplayBuffer:
        def __init__(self):
            self.retired = []

        def sample(self, partition_id, global_steps, expected_uids):
            del global_steps, expected_uids
            return main_ppo_sync.KVBatchMeta(
                partition_id=partition_id,
                keys=[],
                tags=[],
                extra_info={
                    "failed_uids": ["failed"],
                    "failed_output_keys": ["failed_0_0"],
                },
            )

        def retire(self, partition_id, prompt_uids):
            self.retired.append((partition_id, prompt_uids))

    class ClearTQ:
        cleared = []

        @classmethod
        def kv_clear(cls, keys, partition_id):
            cls.cleared.append((partition_id, list(keys)))

    trainer = object.__new__(PPOTrainer)
    trainer.global_steps = 0
    trainer.replay_buffer = FailedReplayBuffer()
    monkeypatch.setattr(main_ppo_sync, "tq", ClearTQ)

    batch = trainer._sample_replay_batch(
        partition_id="val",
        prompt_uids=["failed"],
        require_all_success=False,
    )

    assert batch.keys == []
    assert ClearTQ.cleared == [("val", ["failed", "failed_0_0"])]
    assert trainer.replay_buffer.retired == [("val", ["failed"])]


def test_extract_online_observation_groups(monkeypatch):
    trainer = object.__new__(PPOTrainer)
    trainer.online_sampling_config = OnlineSamplingConfig(
        enabled=True,
        state_path="/tmp/state.sqlite",
        rollout_n=2,
        save_responses=False,
    )
    trainer.global_steps = 4
    trainer.tokenizer = SimpleNamespace()
    monkeypatch.setattr(main_ppo_sync, "tq", FakeTQ)

    groups = trainer._extract_online_observation_groups(
        FakeBatch(),
        phase="train",
        generation_round=2,
    )

    by_sample = {observation.sample_id: (observation, keys) for observation, keys, _ in groups}
    first, first_keys = by_sample["sample-a"]
    second, second_keys = by_sample["sample-b"]
    assert first.rewards == (0.0, 1.0)
    assert first.sigma == 0.5
    assert first.policy_step == 4
    assert first.generation_round == 2
    assert first.attempt_id == "uid-a"
    assert first_keys == ["uid-a_0_0", "uid-a_1_0"]
    assert second.rewards == (1.0, 1.0)
    assert second.sigma == 0.0
    assert second_keys == ["uid-b_0_0", "uid-b_1_0"]


def test_extract_online_observation_groups_rejects_partial_transfer_queue_reads(monkeypatch):
    class PartialFakeTQ(FakeTQ):
        @staticmethod
        def kv_batch_get(keys, partition_id, select_fields):
            data = FakeTQ.kv_batch_get(keys, partition_id, select_fields)
            return {
                key: value[:-1] if isinstance(value, list) else value[:-1]
                for key, value in data.items()
            }

    trainer = object.__new__(PPOTrainer)
    trainer.online_sampling_config = OnlineSamplingConfig(
        enabled=True,
        state_path="/tmp/state.sqlite",
        rollout_n=2,
        save_responses=False,
    )
    trainer.global_steps = 4
    trainer.tokenizer = SimpleNamespace()
    monkeypatch.setattr(main_ppo_sync, "tq", PartialFakeTQ)

    with pytest.raises(RuntimeError, match="returned 3 rows for 4 keys"):
        trainer._extract_online_observation_groups(
            FakeBatch(),
            phase="train",
            generation_round=2,
        )


def test_model_path_fingerprint_changes_with_model_manifest(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"test"}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = _model_path_fingerprint(tmp_path)

    weights.write_bytes(b"second-version")
    second = _model_path_fingerprint(tmp_path)

    assert first["path"] == second["path"]
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_online_sampling_state_semantics_include_ema_and_batch_shape():
    first = _online_sampling_state_semantics(
        OnlineSamplingConfig(enabled=True, state_path="/tmp/state.sqlite", ema_alpha=0.8),
        train_batch_size=128,
        gen_batch_size=160,
    )
    second = _online_sampling_state_semantics(
        OnlineSamplingConfig(enabled=True, state_path="/tmp/state.sqlite", ema_alpha=0.6),
        train_batch_size=128,
        gen_batch_size=160,
    )

    assert first["ema_alpha"] == 0.8
    assert first["train_batch_size"] == 128
    assert first != second
