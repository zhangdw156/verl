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

import hashlib
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.bfcl_v4_agent_loop import (
    BFCLV4MultiTurnAgentLoop,
    _instantiate_handler,
)
from verl.trainer.main_ppo import _embed_bfcl_handler_source


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return bytes(token_ids).decode()


class _QueuedServer:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate(self, request_id, prompt_ids, sampling_params):
        del request_id, prompt_ids, sampling_params
        text = self.responses.pop(0)
        return SimpleNamespace(
            token_ids=list(text.encode()),
            log_probs=None,
            num_preempted=0,
            extra_fields={"global_steps": 3},
        )


class _Handler:
    def _pre_query_processing_prompting(self, entry):
        return {"message": [], "function": entry["function"]}

    def add_first_turn_message_prompting(self, inference_data, messages):
        inference_data["message"].extend(messages)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data, messages):
        inference_data["message"].extend(messages)
        return inference_data

    def _format_prompt(self, messages, functions):
        return f"{messages}|{functions}"

    def _parse_query_response_prompting(self, response):
        return {"model_responses": response.choices[0].text}

    def _add_assistant_message_prompting(self, inference_data, model_response_data):
        inference_data["message"].append({"role": "assistant", "content": model_response_data["model_responses"]})
        return inference_data

    def _add_execution_results_prompting(self, inference_data, execution_results, model_response_data):
        del model_response_data
        inference_data["message"].extend({"role": "tool", "content": item} for item in execution_results)
        return inference_data

    def decode_execute(self, response, has_tool_call_tag=False):
        del has_tool_call_tag
        if response == "CALL":
            return ["normal()"]
        if response == "DONE":
            return []
        raise ValueError("invalid response")


def _make_loop(evaluation_result):
    loop = object.__new__(BFCLV4MultiTurnAgentLoop)
    loop.bfcl_config = {"enabled": True}
    loop.max_steps_per_turn = 20
    loop.handler = _Handler()
    loop.prompt_length = 2048
    loop.response_length = 2048
    loop.tokenizer = _Tokenizer()
    loop.server_manager = _QueuedServer(["CALL", "DONE"])
    runtime_module = SimpleNamespace()
    executed = []

    def execute_calls(calls, *args, **kwargs):
        del args, kwargs
        executed.extend(calls)
        return (["ok"] if calls else []), {}

    def multi_turn_checker(raw_results, ground_truth, entry, category, model_name):
        del ground_truth, entry, category, model_name
        assert raw_results == [[["normal()"]]]
        return evaluation_result

    loop.runtime = SimpleNamespace(
        additional_function_prompt="functions={functions}",
        execute_calls=execute_calls,
        is_empty_response=lambda calls: not calls,
        multi_turn_checker=multi_turn_checker,
        multi_turn_utils=runtime_module,
    )
    return loop, executed


@pytest.mark.asyncio
async def test_agent_loop_executes_and_scores_current_policy():
    loop, executed = _make_loop({"valid": True})
    output = await loop.run(
        {"temperature": 0},
        _verl_validate=True,
        uid="uid",
        bfcl_category="multi_turn_base",
        bfcl_entry={
            "id": "multi_turn_base_0",
            "question": [[{"role": "user", "content": "test"}]],
            "function": [{"name": "normal"}],
            "initial_config": {},
            "involved_classes": [],
        },
        reward_model={"ground_truth": [["normal()"]]},
    )

    assert executed == ["normal()"]
    assert output.reward_score == 1.0
    assert output.extra_fields["reward_extra_info"]["acc"] == 1.0
    assert output.extra_fields["reward_extra_info"]["bfcl_tool_calls"] == 1.0
    assert output.metrics.num_preempted == 0


@pytest.mark.asyncio
async def test_agent_loop_preserves_official_failure_metadata():
    loop, _ = _make_loop(
        {
            "valid": False,
            "error_type": "multi_turn:instance_state_mismatch",
        }
    )
    output = await loop.run(
        {"temperature": 0},
        _verl_validate=True,
        uid="uid",
        bfcl_category="multi_turn_base",
        bfcl_entry={
            "id": "multi_turn_base_0",
            "question": [[{"role": "user", "content": "test"}]],
            "function": [{"name": "normal"}],
            "initial_config": {},
            "involved_classes": [],
        },
        reward_model={"ground_truth": [["normal()"]]},
    )

    assert output.reward_score == 0.0
    assert output.extra_fields["reward_extra_info"]["bfcl_error_type"] == "multi_turn:instance_state_mismatch"


@pytest.mark.asyncio
async def test_unsafe_call_is_rejected_without_execution():
    loop, executed = _make_loop({"valid": True})
    loop.server_manager = _QueuedServer(["UNSAFE"])
    loop.handler.decode_execute = lambda response, has_tool_call_tag=False: ["normal(value=other())"]

    output = await loop.run(
        {"temperature": 0},
        _verl_validate=True,
        uid="uid",
        bfcl_category="multi_turn_base",
        bfcl_entry={
            "id": "multi_turn_base_0",
            "question": [[{"role": "user", "content": "test"}]],
            "function": [{"name": "normal"}],
            "initial_config": {},
            "involved_classes": [],
        },
        reward_model={"ground_truth": [["normal()"]]},
    )

    assert executed == []
    assert output.reward_score == 0.0
    assert output.extra_fields["reward_extra_info"]["bfcl_unsafe_calls"] == 1.0
    assert output.extra_fields["reward_extra_info"]["bfcl_error_type"] == "bfcl:unsafe_function_call"


@pytest.mark.asyncio
async def test_agent_loop_rejects_training_use():
    loop, _ = _make_loop({"valid": True})

    with pytest.raises(RuntimeError, match="validation-only"):
        await loop.run(
            {"temperature": 0},
            uid="uid",
            bfcl_category="multi_turn_base",
            bfcl_entry={},
            reward_model={"ground_truth": []},
        )


def test_independent_handler_file_is_loaded(tmp_path):
    handler_path = tmp_path / "handler.py"
    handler_path.write_text(
        """
class Handler:
    def _pre_query_processing_prompting(self): pass
    def add_first_turn_message_prompting(self): pass
    def _add_next_turn_user_message_prompting(self): pass
    def _format_prompt(self): pass
    def _parse_query_response_prompting(self): pass
    def _add_assistant_message_prompting(self): pass
    def _add_execution_results_prompting(self): pass
    def decode_execute(self): pass
""",
        encoding="utf-8",
    )

    handler = _instantiate_handler(OmegaConf.create({"path": str(handler_path), "name": "Handler", "kwargs": {}}))
    assert handler.__class__.__name__ == "Handler"


def test_handler_contract_fails_closed(tmp_path):
    handler_path = tmp_path / "handler.py"
    handler_path.write_text("class Handler: pass\n", encoding="utf-8")

    with pytest.raises(TypeError, match="missing required methods"):
        _instantiate_handler(OmegaConf.create({"path": str(handler_path), "name": "Handler", "kwargs": {}}))


def test_embedded_handler_source_does_not_require_worker_file():
    source = """
from dataclasses import dataclass

@dataclass
class Handler:
    def _pre_query_processing_prompting(self): pass
    def add_first_turn_message_prompting(self): pass
    def _add_next_turn_user_message_prompting(self): pass
    def _format_prompt(self): pass
    def _parse_query_response_prompting(self): pass
    def _add_assistant_message_prompting(self): pass
    def _add_execution_results_prompting(self): pass
    def decode_execute(self): pass
"""
    handler = _instantiate_handler(
        OmegaConf.create(
            {
                "path": "/driver/only/handler.py",
                "name": "Handler",
                "kwargs": {},
                "source": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    )
    assert handler.__class__.__name__ == "Handler"


def test_driver_embeds_handler_source(tmp_path):
    handler_path = tmp_path / "handler.py"
    handler_path.write_text("class Handler: pass\n", encoding="utf-8")
    config = OmegaConf.create(
        {
            "data": {
                "bfcl_v4": {
                    "enabled": True,
                    "handler": {
                        "path": str(handler_path),
                        "name": "Handler",
                        "kwargs": {},
                        "source": None,
                        "source_sha256": None,
                    },
                }
            }
        }
    )

    _embed_bfcl_handler_source(config)

    assert config.data.bfcl_v4.handler.source == "class Handler: pass\n"
    assert (
        config.data.bfcl_v4.handler.source_sha256
        == hashlib.sha256(config.data.bfcl_v4.handler.source.encode()).hexdigest()
    )
