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

from omegaconf import OmegaConf

import verl.trainer.main_ppo as main_ppo


class _RemoteMethod:
    def remote(self, config):
        return config


class _Runner:
    run = _RemoteMethod()


class _TaskRunnerClass:
    @staticmethod
    def remote():
        return _Runner()


def test_run_ppo_resolves_ray_runtime_env_before_ray_init(monkeypatch, tmp_path):
    config = OmegaConf.create(
        {
            "verl_meta_config": {
                "work_root": str(tmp_path),
                "venv_verl": "/runtime/verl",
            },
            "data": {"bfcl_v4": {"enabled": False}},
            "transfer_queue": {"enable": True},
            "ray_kwargs": {
                "ray_init": {
                    "runtime_env": {
                        "env_vars": {
                            "PATH": "${verl_meta_config.venv_verl}/bin:/usr/bin",
                            "VIRTUAL_ENV": "${verl_meta_config.venv_verl}",
                        },
                        "working_dir": "${verl_meta_config.work_root}",
                    }
                },
                "timeline_json_file": None,
            },
            "global_profiler": {
                "tool": None,
                "steps": None,
            },
        }
    )
    captured = {}

    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(main_ppo.ray, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(main_ppo.ray, "get", lambda value: value)
    monkeypatch.setattr(main_ppo, "get_ppo_ray_runtime_env", lambda: {"env_vars": {}})
    monkeypatch.setattr(main_ppo, "is_cuda_available", False)

    main_ppo.run_ppo(config, task_runner_class=_TaskRunnerClass)

    runtime_env = captured["runtime_env"]
    assert runtime_env["working_dir"] == str(tmp_path)
    assert runtime_env["env_vars"]["PATH"] == "/runtime/verl/bin:/usr/bin"
    assert runtime_env["env_vars"]["VIRTUAL_ENV"] == "/runtime/verl"
    assert runtime_env["env_vars"]["TRANSFER_QUEUE_ENABLE"] == "1"
