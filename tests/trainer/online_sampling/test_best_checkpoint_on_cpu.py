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

import json
import shutil

import pytest
from omegaconf import OmegaConf

from verl.trainer.main_ppo_sync import PPOTrainer, _parse_transfer_queue_key


class FakeCheckpointManager:
    def __init__(self):
        self.sleep_count = 0
        self.wake_count = 0

    def sleep_replicas(self):
        self.sleep_count += 1

    def wake_up_replicas(self):
        self.wake_count += 1


class FakeOnlineController:
    def __init__(self):
        self.snapshots = []
        self.deleted_snapshots = []

    def save_checkpoint_snapshot(self, kind, step):
        self.snapshots.append((kind, step))

    def delete_checkpoint_snapshot(self, kind, step):
        self.deleted_snapshots.append((kind, step))


def make_trainer(tmp_path):
    trainer = object.__new__(PPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "checkpoint": {
                        "async_save": False,
                    }
                }
            },
            "critic": {
                "checkpoint": {
                    "async_save": False,
                }
            },
            "trainer": {
                "default_local_dir": str(tmp_path),
                "best_checkpoint": {
                    "enabled": True,
                    "metric": "val/score",
                    "mode": "max",
                    "min_delta": 0.0,
                    "directory": "best_checkpoint",
                },
            },
        }
    )
    trainer.global_steps = 1
    trainer.best_checkpoint_value = None
    trainer.best_checkpoint_step = None
    trainer.use_critic = False
    trainer.online_sampling_controller = FakeOnlineController()
    trainer.checkpoint_manager = FakeCheckpointManager()
    trainer.regular_save_calls = []

    def save_regular_checkpoint():
        step_folder = tmp_path / f"global_step_{trainer.global_steps}"
        actor_folder = step_folder / "actor"
        actor_folder.mkdir(parents=True, exist_ok=True)
        (actor_folder / "step.txt").write_text(str(trainer.global_steps), encoding="utf-8")
        (step_folder / "data.pt").write_bytes(b"data")
        (tmp_path / "latest_checkpointed_iteration.txt").write_text(
            str(trainer.global_steps),
            encoding="utf-8",
        )
        trainer.regular_save_calls.append(trainer.global_steps)

    trainer._save_checkpoint = save_regular_checkpoint
    return trainer


def test_best_checkpoint_updates_only_on_improvement(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer._maybe_save_best_checkpoint({"val/score": 0.5})
    trainer.global_steps = 2
    trainer._maybe_save_best_checkpoint({"val/score": 0.4})

    best_root = tmp_path / "best_checkpoint"
    assert (best_root / "best_checkpointed_iteration.txt").read_text() == "1"
    metadata = json.loads((best_root / "global_step_1/best_metric.json").read_text())
    assert metadata["value"] == 0.5
    assert not (best_root / "global_step_2").exists()
    assert trainer.online_sampling_controller.snapshots == [("best", 1)]
    assert trainer.regular_save_calls == [1]
    assert trainer.checkpoint_manager.sleep_count == 1
    assert trainer.checkpoint_manager.wake_count == 1


def test_best_checkpoint_replaces_previous_after_complete_save(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._maybe_save_best_checkpoint({"val/score": 0.5})
    trainer.global_steps = 3

    trainer._maybe_save_best_checkpoint({"val/score": 0.7})

    best_root = tmp_path / "best_checkpoint"
    assert (best_root / "best_checkpointed_iteration.txt").read_text() == "3"
    assert not (best_root / "global_step_1").exists()
    assert (best_root / "global_step_3/actor/step.txt").read_text() == "3"
    assert trainer.online_sampling_controller.snapshots == [
        ("best", 1),
        ("best", 3),
    ]
    assert trainer.online_sampling_controller.deleted_snapshots == [("best", 1)]
    shutil.rmtree(tmp_path / "global_step_3")
    assert (best_root / "global_step_3/actor/step.txt").read_text() == "3"


def test_best_checkpoint_reuses_complete_regular_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_checkpoint()
    trainer.regular_save_calls.clear()

    trainer._maybe_save_best_checkpoint({"val/score": 0.5})

    assert trainer.regular_save_calls == []
    assert (tmp_path / "best_checkpoint/global_step_1/actor/step.txt").read_text() == "1"


def test_best_checkpoint_rejects_async_checkpoint_saving(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config.actor_rollout_ref.actor.checkpoint.async_save = True

    with pytest.raises(ValueError, match="asynchronous checkpoint"):
        trainer._maybe_save_best_checkpoint({"val/score": 0.5})


def test_best_checkpoint_rejects_torchtitan_layout(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.config.actor_rollout_ref.actor.torchtitan = {}

    with pytest.raises(ValueError, match="TorchTitan"):
        trainer._maybe_save_best_checkpoint({"val/score": 0.5})


def test_best_checkpoint_rejects_torchtitan_critic_layout(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.use_critic = True
    trainer.config.critic.torchtitan = {}

    with pytest.raises(ValueError, match="TorchTitan"):
        trainer._maybe_save_best_checkpoint({"val/score": 0.5})


def test_best_checkpoint_fails_closed_when_metric_is_missing(tmp_path):
    trainer = make_trainer(tmp_path)

    with pytest.raises(KeyError, match="val/score"):
        trainer._maybe_save_best_checkpoint({"val/other": 1.0})


def test_transfer_queue_key_parser_uses_rightmost_fields():
    assert _parse_transfer_queue_key("uid_with_underscores_7_2") == (
        "uid_with_underscores",
        7,
        2,
    )
