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

import ast
import copy
import hashlib
import re
import sys
import time
from functools import lru_cache
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.utils.import_utils import load_extern_object

_REQUIRED_HANDLER_METHODS = (
    "_pre_query_processing_prompting",
    "add_first_turn_message_prompting",
    "_add_next_turn_user_message_prompting",
    "_format_prompt",
    "_parse_query_response_prompting",
    "_add_assistant_message_prompting",
    "_add_execution_results_prompting",
    "decode_execute",
)


def _require_bfcl_runtime():
    try:
        from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
            is_empty_execute_response,
        )
    except ImportError as exc:
        raise ImportError(
            "BFCLv4 validation requires the `bfcl_eval` package on every Ray worker. "
            "Install the Berkeley Function Calling Leaderboard package before enabling data.bfcl_v4."
        ) from exc

    return SimpleNamespace(
        additional_function_prompt=DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
        execute_calls=execute_multi_turn_func_call,
        is_empty_response=is_empty_execute_response,
        multi_turn_checker=multi_turn_checker,
        multi_turn_utils=multi_turn_utils,
    )


@lru_cache(maxsize=32)
def _load_handler_object(path: str, name: str):
    return load_extern_object(path, name)


@lru_cache(maxsize=32)
def _load_handler_object_from_source(source_sha256: str, source: str, name: str, filename: str):
    actual_sha256 = hashlib.sha256(source.encode()).hexdigest()
    if actual_sha256 != source_sha256:
        raise ValueError("Embedded BFCL handler source checksum does not match data.bfcl_v4.handler.source_sha256")
    module_name = f"verl_bfcl_handler_{source_sha256[:16]}"
    module = ModuleType(module_name)
    module.__file__ = filename
    sys.modules[module_name] = module
    try:
        exec(compile(source, filename, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(module, name):
        raise AttributeError(f"Object `{name}` was not found in embedded BFCL handler `{filename}`")
    return getattr(module, name)


def _instantiate_handler(handler_config):
    path = handler_config.get("path")
    if not path:
        raise ValueError("data.bfcl_v4.handler.path must point to an independent Python handler file")
    name = handler_config.get("name", "Handler")
    raw_kwargs = handler_config.get("kwargs", {})
    kwargs = OmegaConf.to_container(raw_kwargs, resolve=True) if OmegaConf.is_config(raw_kwargs) else dict(raw_kwargs)
    source = handler_config.get("source")
    source_sha256 = handler_config.get("source_sha256")
    if source:
        if not source_sha256:
            raise ValueError("Embedded BFCL handler source is missing source_sha256")
        handler_object = _load_handler_object_from_source(source_sha256, source, name, path)
    else:
        resolved_path = resolve_config_path(path)
        handler_object = _load_handler_object(resolved_path, name)

    if isinstance(handler_object, type):
        handler = handler_object(**kwargs)
    elif callable(handler_object):
        handler = handler_object(**kwargs)
    else:
        if kwargs:
            raise TypeError("Handler kwargs cannot be used when the exported handler object is already instantiated")
        handler = handler_object

    missing_methods = [method for method in _REQUIRED_HANDLER_METHODS if not callable(getattr(handler, method, None))]
    if missing_methods:
        raise TypeError(
            f"BFCL handler `{name}` from `{path}` is missing required methods: {', '.join(missing_methods)}"
        )
    return handler


def _is_safe_argument(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_is_safe_argument(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_safe_argument(key) and _is_safe_argument(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        return _is_safe_argument(node.operand)
    return False


def _validate_function_calls(function_calls: list[str], allowed_functions: set[str]) -> bool:
    for function_call in function_calls:
        if not isinstance(function_call, str):
            return False
        try:
            expression = ast.parse(function_call, mode="eval").body
        except SyntaxError:
            return False
        if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
            return False
        if expression.func.id not in allowed_functions:
            return False
        if any(not _is_safe_argument(argument) for argument in expression.args):
            return False
        if any(keyword.arg is None or not _is_safe_argument(keyword.value) for keyword in expression.keywords):
            return False
    return True


def _make_api_response(text: str, prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _extract_error_type(evaluation_result: dict[str, Any]) -> str:
    error = evaluation_result.get("error")
    if isinstance(error, dict):
        error_type = error.get("error_type")
        if isinstance(error_type, str):
            return error_type
    error_type = evaluation_result.get("error_type")
    return error_type if isinstance(error_type, str) else ""


def _evaluate_single_multi_turn_entry(
    entry_id: str,
    decoded_model_results: list[list[list[str]]],
    ground_truth_list: list[list[str]],
    prompt_entry: dict[str, Any],
    model_name: str,
    test_category: str,
    runtime,
) -> dict[str, Any]:
    if not isinstance(decoded_model_results, list):
        return {
            "valid": False,
            "error": {"error_type": "multi_turn:inference_error"},
        }
    if len(decoded_model_results) != len(ground_truth_list):
        return {
            "valid": False,
            "error": {"error_type": "multi_turn:force_terminated"},
        }

    checker_result = runtime.multi_turn_checker(
        decoded_model_results,
        ground_truth_list,
        copy.deepcopy(prompt_entry),
        test_category,
        model_name,
    )
    if checker_result["valid"]:
        return {"valid": True}
    return {
        "id": entry_id,
        "valid": False,
        "error": {key: value for key, value in checker_result.items() if key != "valid"},
    }


def _cleanup_bfcl_instances(runtime_module, prefixes: tuple[str, ...], entry_id: str) -> None:
    sanitized_entry_id = re.sub(r"[-./:]", "_", entry_id)
    sanitized_prefixes = tuple(re.sub(r"[-./:]", "_", prefix) for prefix in prefixes)
    for name in list(vars(runtime_module)):
        if (
            name.endswith("_instance")
            and f"_{sanitized_entry_id}_" in name
            and any(name.startswith(prefix) for prefix in sanitized_prefixes)
        ):
            vars(runtime_module).pop(name, None)


@register("bfcl_v4_multi_turn")
class BFCLV4MultiTurnAgentLoop(AgentLoopBase):
    """Run the official BFCLv4 multi-turn environment inside VeRL validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bfcl_config = self.data_config.get("bfcl_v4", {})
        if not self.bfcl_config.get("enabled", False):
            raise ValueError("BFCLV4MultiTurnAgentLoop requires data.bfcl_v4.enabled=true")
        self.max_steps_per_turn = int(self.bfcl_config.get("max_steps_per_turn", 20))
        self.handler = _instantiate_handler(self.bfcl_config.get("handler", {}))
        self.runtime = _require_bfcl_runtime()
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        if self.rollout_config.val_kwargs.n != 1:
            raise ValueError("Official BFCLv4 validation requires actor_rollout_ref.rollout.val_kwargs.n=1")
        if self.rollout_config.val_kwargs.do_sample:
            raise ValueError("Official BFCLv4 validation requires actor_rollout_ref.rollout.val_kwargs.do_sample=false")

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if not kwargs.pop("_verl_validate", False):
            raise RuntimeError("BFCLV4MultiTurnAgentLoop is validation-only and cannot be used for training rollout")
        entry = copy.deepcopy(kwargs["bfcl_entry"])
        score_entry = copy.deepcopy(entry)
        ground_truth = copy.deepcopy(kwargs["reward_model"]["ground_truth"])
        category = kwargs["bfcl_category"]
        entry_id = entry["id"]
        uid = str(kwargs.get("uid", uuid4().hex))
        request_id = f"bfcl-v4-{uid}"
        runtime_name = f"verl_runtime_{uid}"
        score_name = f"verl_score_{uid}"

        metrics = AgentLoopMetrics()
        last_prompt_ids = None
        last_response_ids: list[int] = []
        last_response_logprobs: list[float] | None = None
        raw_model_results: list[list[str]] = []
        decoded_model_results: list[list[list[str]]] = []
        force_terminated = False
        prompt_too_long = False
        decode_errors = 0
        tool_call_count = 0
        unsafe_call_count = 0
        assistant_turns = 0
        user_turns = 0
        tool_turns = 0
        min_global_steps = None
        max_global_steps = None

        try:
            initial_config = entry.get("initial_config", {})
            involved_classes = entry["involved_classes"]
            holdout_function = entry.get("missed_function", {})
            long_context = "long_context" in category or "composite" in category
            allowed_functions = {function_doc["name"] for function_doc in entry["function"]}
            for held_out_functions in holdout_function.values():
                allowed_functions.update(function_doc["name"] for function_doc in held_out_functions)

            self.runtime.execute_calls(
                [],
                initial_config,
                involved_classes,
                runtime_name,
                entry_id,
                long_context=long_context,
                is_evaL_run=False,
            )

            inference_data = self.handler._pre_query_processing_prompting(entry)
            for turn_index, original_turn_message in enumerate(entry["question"]):
                current_turn_message = copy.deepcopy(original_turn_message)
                if str(turn_index) in holdout_function:
                    if current_turn_message:
                        raise ValueError(
                            f"BFCL holdout turn {turn_index} for `{entry_id}` must have an empty user turn"
                        )
                    current_turn_message = [
                        {
                            "role": "user",
                            "content": self.runtime.additional_function_prompt.format(
                                functions=holdout_function[str(turn_index)]
                            ),
                        }
                    ]

                if turn_index == 0:
                    inference_data = self.handler.add_first_turn_message_prompting(inference_data, current_turn_message)
                else:
                    inference_data = self.handler._add_next_turn_user_message_prompting(
                        inference_data, current_turn_message
                    )
                user_turns += len(current_turn_message)

                current_turn_results = []
                current_turn_decoded = []
                step_count = 0
                while True:
                    prompt_text = self.handler._format_prompt(inference_data["message"], inference_data["function"])
                    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                    if len(prompt_ids) > self.prompt_length:
                        last_prompt_ids = list(prompt_ids[-self.prompt_length :])
                        last_response_ids = []
                        last_response_logprobs = None
                        prompt_too_long = True
                        force_terminated = True
                        break
                    started = time.perf_counter()
                    generation = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                    )
                    metrics.generate_sequences += time.perf_counter() - started
                    if generation.num_preempted is not None:
                        if metrics.num_preempted < 0:
                            metrics.num_preempted = 0
                        metrics.num_preempted += generation.num_preempted

                    generated_ids = list(generation.token_ids)
                    last_prompt_ids = list(prompt_ids)
                    last_response_ids = generated_ids
                    last_response_logprobs = list(generation.log_probs) if generation.log_probs is not None else None

                    global_steps = generation.extra_fields.get("global_steps")
                    if global_steps is not None:
                        min_global_steps = (
                            global_steps if min_global_steps is None else min(min_global_steps, global_steps)
                        )
                        max_global_steps = (
                            global_steps if max_global_steps is None else max(max_global_steps, global_steps)
                        )

                    response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    api_response = _make_api_response(response_text, len(prompt_ids), len(generated_ids))
                    model_response_data = self.handler._parse_query_response_prompting(api_response)
                    model_responses = model_response_data["model_responses"]
                    inference_data = self.handler._add_assistant_message_prompting(inference_data, model_response_data)

                    current_turn_results.append(model_responses)
                    assistant_turns += 1
                    try:
                        decoded_responses = self.handler.decode_execute(model_responses, has_tool_call_tag=False)
                    except Exception:
                        decode_errors += 1
                        break

                    model_response_data["model_responses_decoded"] = decoded_responses
                    if self.runtime.is_empty_response(decoded_responses):
                        break
                    if not _validate_function_calls(decoded_responses, allowed_functions):
                        unsafe_call_count += 1
                        break
                    current_turn_decoded.append(decoded_responses)

                    started = time.perf_counter()
                    execution_results, _ = self.runtime.execute_calls(
                        decoded_responses,
                        initial_config,
                        involved_classes,
                        runtime_name,
                        entry_id,
                        long_context=long_context,
                        is_evaL_run=False,
                    )
                    metrics.tool_calls += time.perf_counter() - started
                    tool_call_count += len(decoded_responses)
                    tool_turns += len(execution_results)
                    inference_data = self.handler._add_execution_results_prompting(
                        inference_data, execution_results, model_response_data
                    )

                    step_count += 1
                    if step_count > self.max_steps_per_turn:
                        force_terminated = True
                        break

                raw_model_results.append(current_turn_results)
                decoded_model_results.append(current_turn_decoded)
                if force_terminated:
                    break

            if prompt_too_long:
                evaluation_result = {
                    "valid": False,
                    "error": {"error_type": "bfcl:prompt_too_long"},
                }
            elif unsafe_call_count:
                evaluation_result = {
                    "valid": False,
                    "error": {"error_type": "bfcl:unsafe_function_call"},
                }
            else:
                started = time.perf_counter()
                evaluation_result = _evaluate_single_multi_turn_entry(
                    entry_id,
                    decoded_model_results,
                    ground_truth,
                    score_entry,
                    score_name,
                    category,
                    self.runtime,
                )
                metrics.compute_score += time.perf_counter() - started
            accuracy = float(bool(evaluation_result["valid"]))
            error_type = _extract_error_type(evaluation_result)
        finally:
            _cleanup_bfcl_instances(
                self.runtime.multi_turn_utils,
                (runtime_name, score_name),
                entry_id,
            )

        if last_prompt_ids is None:
            raise ValueError(f"BFCL entry `{entry_id}` did not produce a model prompt")
        if not last_response_ids:
            fallback_token = self.tokenizer.eos_token_id
            if fallback_token is None:
                fallback_token = self.tokenizer.pad_token_id or 0
            last_response_ids = [fallback_token]
            last_response_logprobs = None

        last_response_ids = last_response_ids[: self.response_length]
        response_mask = [1] * len(last_response_ids)
        if last_response_logprobs is not None:
            last_response_logprobs = last_response_logprobs[: self.response_length]

        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
            "extras": {
                "bfcl_id": entry_id,
                "bfcl_category": category,
                "bfcl_error_type": error_type,
            },
            "reward_extra_info": {
                "acc": accuracy,
                "bfcl_force_terminated": float(force_terminated),
                "bfcl_turns_completed": float(len(raw_model_results)),
                "bfcl_tool_calls": float(tool_call_count),
                "bfcl_decode_errors": float(decode_errors),
                "bfcl_unsafe_calls": float(unsafe_call_count),
                "bfcl_prompt_too_long": float(prompt_too_long),
                "bfcl_error_type": error_type,
            },
        }

        return AgentLoopOutput(
            prompt_ids=last_prompt_ids,
            response_ids=last_response_ids,
            response_mask=response_mask,
            response_logprobs=last_response_logprobs,
            reward_score=accuracy,
            num_turns=user_turns + assistant_turns + tool_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )
