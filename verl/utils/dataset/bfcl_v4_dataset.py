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

import copy
import json
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

BFCL_V4_MULTI_TURN_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)
BFCL_V4_DATA_SOURCE_PREFIX = "bfcl_v4/"


def _require_bfcl_runtime():
    try:
        from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
        from bfcl_eval.utils import add_language_specific_hint_to_function_doc
    except ImportError as exc:
        raise ImportError(
            "BFCLv4 validation requires the `bfcl_eval` package on every Ray worker. "
            "Install the Berkeley Function Calling Leaderboard package before enabling data.bfcl_v4."
        ) from exc

    return MULTI_TURN_FUNC_DOC_FILE_MAPPING, add_language_specific_hint_to_function_doc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_number}") from exc
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(f"Missing or invalid `id` in {path}:{line_number}")
            if row_id in seen_ids:
                raise ValueError(f"Duplicate BFCL id `{row_id}` in {path}")
            seen_ids.add(row_id)
            rows.append(row)
    return rows


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}") from None
            rows.append(row)
        return rows
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"Expected a JSON object, list, or JSONL records in {path}")


def _resolve_data_root(data_files: str | list[str], config: DictConfig) -> Path:
    bfcl_config = config.get("bfcl_v4", {})
    configured_root = bfcl_config.get("data_root")
    if configured_root:
        root = Path(configured_root).expanduser()
    else:
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        if len(data_files) != 1:
            raise ValueError(
                "BFCLv4 validation expects one raw data root in data.val_files or an explicit data.bfcl_v4.data_root."
            )
        root = Path(data_files[0]).expanduser()

    candidates = (
        root,
        root / "data",
        root / "bfcl_eval" / "data",
    )
    for candidate in candidates:
        if (candidate / "BFCL_v4_multi_turn_base.json").is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate BFCLv4 multi-turn data. Expected "
        "`BFCL_v4_multi_turn_base.json` under the configured root, `root/data`, "
        "or `root/bfcl_eval/data`."
    )


class BFCLV4MultiTurnDataset(Dataset):
    """Load the raw BFCLv4 multi-turn suite directly for VeRL validation."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        del tokenizer, processor
        self.config = config
        self.data_root = _resolve_data_root(data_files, config)
        self.bfcl_config = config.get("bfcl_v4", {})
        self.categories = tuple(self.bfcl_config.get("categories", BFCL_V4_MULTI_TURN_CATEGORIES))
        self.strict = bool(self.bfcl_config.get("strict", True))
        self.expected_examples_per_category = int(self.bfcl_config.get("expected_examples_per_category", 200))
        self.rows = self._load_rows()

        if max_samples > 0 and max_samples < len(self.rows):
            if self.strict:
                raise ValueError(
                    "Official BFCLv4 validation cannot use data.val_max_samples. "
                    "Set data.bfcl_v4.strict=false for a non-official smoke subset."
                )
            self.rows = self._stratified_subset(self.rows, max_samples)

    def _load_rows(self) -> list[dict[str, Any]]:
        function_doc_mapping, add_language_hint = _require_bfcl_runtime()
        function_doc_cache: dict[str, list[dict[str, Any]]] = {}
        result = []
        all_ids = set()

        for category in self.categories:
            prompt_path = self.data_root / f"BFCL_v4_{category}.json"
            answer_path = self.data_root / "possible_answer" / f"BFCL_v4_{category}.json"
            if not prompt_path.is_file():
                raise FileNotFoundError(f"Missing BFCL prompt file: {prompt_path}")
            if not answer_path.is_file():
                raise FileNotFoundError(f"Missing BFCL ground-truth file: {answer_path}")

            prompt_rows = _read_jsonl(prompt_path)
            answer_rows = _read_jsonl(answer_path)
            if self.strict and len(prompt_rows) != self.expected_examples_per_category:
                raise ValueError(
                    f"{prompt_path.name} contains {len(prompt_rows)} examples; "
                    f"expected {self.expected_examples_per_category}."
                )

            answers_by_id = {row["id"]: row for row in answer_rows}
            prompt_ids = {row["id"] for row in prompt_rows}
            if prompt_ids != set(answers_by_id):
                missing_answers = sorted(prompt_ids - set(answers_by_id))
                extra_answers = sorted(set(answers_by_id) - prompt_ids)
                raise ValueError(
                    f"BFCL prompt/ground-truth IDs do not match for {category}: "
                    f"missing_answers={missing_answers[:5]}, extra_answers={extra_answers[:5]}"
                )

            for prompt_row in prompt_rows:
                entry_id = prompt_row["id"]
                if entry_id in all_ids:
                    raise ValueError(f"Duplicate BFCL id across categories: {entry_id}")
                all_ids.add(entry_id)

                entry = copy.deepcopy(prompt_row)
                entry["function"] = []
                for class_name in entry.get("involved_classes", []):
                    try:
                        function_doc_file = function_doc_mapping[class_name]
                    except KeyError as exc:
                        raise ValueError(f"No BFCL function-document mapping for class `{class_name}`") from exc
                    if function_doc_file not in function_doc_cache:
                        function_doc_path = self.data_root / "multi_turn_func_doc" / function_doc_file
                        if not function_doc_path.is_file():
                            raise FileNotFoundError(f"Missing BFCL function-document file: {function_doc_path}")
                        function_doc_cache[function_doc_file] = _read_json_or_jsonl(function_doc_path)
                    entry["function"].extend(copy.deepcopy(function_doc_cache[function_doc_file]))

                self._apply_missed_function_holdout(entry)
                entry = add_language_hint([entry])[0]

                ground_truth = copy.deepcopy(answers_by_id[entry_id]["ground_truth"])
                questions = entry.get("question")
                if not isinstance(questions, list) or not questions:
                    raise ValueError(f"BFCL entry `{entry_id}` has no multi-turn question list")
                if len(questions) != len(ground_truth):
                    raise ValueError(
                        f"BFCL entry `{entry_id}` has {len(questions)} question turns "
                        f"but {len(ground_truth)} ground-truth turns"
                    )

                first_turn = copy.deepcopy(questions[0])
                if not first_turn:
                    first_turn = [{"role": "user", "content": ""}]

                result.append(
                    {
                        "data_source": f"{BFCL_V4_DATA_SOURCE_PREFIX}{category}",
                        "agent_name": "bfcl_v4_multi_turn",
                        "raw_prompt": first_turn,
                        "reward_model": {"style": "rule", "ground_truth": ground_truth},
                        "extra_info": {"index": len(result), "sample_id": entry_id},
                        "index": len(result),
                        "bfcl_entry": entry,
                        "bfcl_category": category,
                        "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
                    }
                )

        return result

    @staticmethod
    def _apply_missed_function_holdout(entry: dict[str, Any]) -> None:
        missed_function = entry.get("missed_function")
        if not missed_function:
            return

        available_functions = entry["function"]
        for turn_index, function_names in list(missed_function.items()):
            held_out = []
            for function_name in function_names:
                match_index = next(
                    (
                        index
                        for index, function_doc in enumerate(available_functions)
                        if function_doc["name"] == function_name
                    ),
                    None,
                )
                if match_index is None:
                    raise ValueError(
                        f"Missed function `{function_name}` for entry `{entry['id']}` "
                        "was not found in the involved function documents"
                    )
                held_out.append(available_functions.pop(match_index))
            missed_function[turn_index] = held_out

    @staticmethod
    def _stratified_subset(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
        by_category = {category: [] for category in BFCL_V4_MULTI_TURN_CATEGORIES}
        for row in rows:
            by_category[row["bfcl_category"]].append(row)

        selected = []
        category_index = 0
        categories = [category for category, category_rows in by_category.items() if category_rows]
        while len(selected) < max_samples and categories:
            category = categories[category_index % len(categories)]
            if by_category[category]:
                selected.append(by_category[category].pop(0))
            else:
                categories.remove(category)
                category_index -= 1
            category_index += 1
        return selected

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return copy.deepcopy(self.rows[item])
