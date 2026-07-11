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

import json

import pytest
from omegaconf import OmegaConf

from verl.utils.dataset.bfcl_v4_dataset import (
    BFCL_V4_MULTI_TURN_CATEGORIES,
    BFCLV4MultiTurnDataset,
)
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


@pytest.fixture
def bfcl_root(tmp_path):
    data_root = tmp_path / "data"
    function_docs = [
        {
            "name": "normal",
            "description": "normal",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "held",
            "description": "held",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    (data_root / "multi_turn_func_doc").mkdir(parents=True)
    (data_root / "multi_turn_func_doc" / "fake_api.json").write_text(json.dumps(function_docs), encoding="utf-8")

    for category in BFCL_V4_MULTI_TURN_CATEGORIES:
        entry_id = f"{category}_0"
        is_miss_func = category == "multi_turn_miss_func"
        prompt = {
            "id": entry_id,
            "question": [[]] if is_miss_func else [[{"role": "user", "content": category}]],
            "initial_config": {"FakeAPI": {}},
            "path": ["FakeAPI.normal"],
            "involved_classes": ["FakeAPI"],
            "excluded_function": [],
        }
        if is_miss_func:
            prompt["missed_function"] = {"0": ["held"]}
        _write_jsonl(data_root / f"BFCL_v4_{category}.json", [prompt])
        _write_jsonl(
            data_root / "possible_answer" / f"BFCL_v4_{category}.json",
            [{"id": entry_id, "ground_truth": [["normal()"]]}],
        )
    return data_root


def _config(data_root):
    return OmegaConf.create(
        {
            "bfcl_v4": {
                "enabled": True,
                "data_root": str(data_root),
                "strict": True,
                "expected_examples_per_category": 1,
                "categories": list(BFCL_V4_MULTI_TURN_CATEGORIES),
            }
        }
    )


def test_loads_raw_four_category_suite(monkeypatch, bfcl_root):
    monkeypatch.setattr(
        "verl.utils.dataset.bfcl_v4_dataset._require_bfcl_runtime",
        lambda: (
            {"FakeAPI": "fake_api.json"},
            lambda entries: entries,
        ),
    )

    dataset = BFCLV4MultiTurnDataset(str(bfcl_root), tokenizer=None, config=_config(bfcl_root))

    assert len(dataset) == 4
    assert {row["bfcl_category"] for row in dataset.rows} == set(BFCL_V4_MULTI_TURN_CATEGORIES)
    assert {row["data_source"] for row in dataset.rows} == {
        f"bfcl_v4/{category}" for category in BFCL_V4_MULTI_TURN_CATEGORIES
    }
    miss_func = next(row for row in dataset.rows if row["bfcl_category"] == "multi_turn_miss_func")
    assert [item["name"] for item in miss_func["bfcl_entry"]["function"]] == ["normal"]
    assert miss_func["bfcl_entry"]["missed_function"]["0"][0]["name"] == "held"


def test_fails_closed_on_missing_category(monkeypatch, bfcl_root):
    monkeypatch.setattr(
        "verl.utils.dataset.bfcl_v4_dataset._require_bfcl_runtime",
        lambda: (
            {"FakeAPI": "fake_api.json"},
            lambda entries: entries,
        ),
    )
    (bfcl_root / "BFCL_v4_multi_turn_base.json").unlink()

    with pytest.raises(FileNotFoundError, match="multi_turn_base"):
        BFCLV4MultiTurnDataset(str(bfcl_root), tokenizer=None, config=_config(bfcl_root))


def test_fails_closed_on_ground_truth_mismatch(monkeypatch, bfcl_root):
    monkeypatch.setattr(
        "verl.utils.dataset.bfcl_v4_dataset._require_bfcl_runtime",
        lambda: (
            {"FakeAPI": "fake_api.json"},
            lambda entries: entries,
        ),
    )
    _write_jsonl(
        bfcl_root / "possible_answer" / "BFCL_v4_multi_turn_base.json",
        [{"id": "wrong_id", "ground_truth": [["normal()"]]}],
    )

    with pytest.raises(ValueError, match="IDs do not match"):
        BFCLV4MultiTurnDataset(str(bfcl_root), tokenizer=None, config=_config(bfcl_root))


def test_validation_dataset_factory_selects_bfcl(monkeypatch, bfcl_root):
    config = _config(bfcl_root)
    assert get_dataset_class(config, is_train=False) is BFCLV4MultiTurnDataset
    assert get_dataset_class(config, is_train=True) is RLHFDataset


def test_strict_validation_rejects_max_samples(monkeypatch, bfcl_root):
    monkeypatch.setattr(
        "verl.utils.dataset.bfcl_v4_dataset._require_bfcl_runtime",
        lambda: (
            {"FakeAPI": "fake_api.json"},
            lambda entries: entries,
        ),
    )

    with pytest.raises(ValueError, match="cannot use data.val_max_samples"):
        BFCLV4MultiTurnDataset(
            str(bfcl_root),
            tokenizer=None,
            config=_config(bfcl_root),
            max_samples=2,
        )
