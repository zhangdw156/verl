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

import pytest

from verl.trainer.ppo.metric_utils import compute_bfcl_v4_multi_turn_metrics


def test_computes_official_unweighted_overall_accuracy():
    category_values = {
        "multi_turn_base": 1.0,
        "multi_turn_miss_func": 0.0,
        "multi_turn_miss_param": 0.5,
        "multi_turn_long_context": 1.0,
    }
    data_sources = []
    accuracies = []
    for category, value in category_values.items():
        data_sources.extend([f"bfcl_v4/{category}"] * 200)
        accuracies.extend([value] * 200)
    result = compute_bfcl_v4_multi_turn_metrics(data_sources, {"acc": accuracies})

    assert result["val-core/bfcl_v4_multi_turn/overall/accuracy"] == pytest.approx(0.625)
    assert result["val-core/bfcl_v4_multi_turn/multi_turn_base/accuracy"] == 1.0


def test_subset_is_labeled_partial():
    data_sources = [
        "bfcl_v4/multi_turn_base",
        "bfcl_v4/multi_turn_miss_func",
        "bfcl_v4/multi_turn_miss_param",
        "bfcl_v4/multi_turn_long_context",
    ]
    result = compute_bfcl_v4_multi_turn_metrics(data_sources, {"acc": [1.0, 0.0, 0.5, 1.0]})

    assert "val-core/bfcl_v4_multi_turn/overall/accuracy" not in result
    assert result["val-aux/bfcl_v4_multi_turn/partial/accuracy"] == pytest.approx(0.625)
    assert result["val-aux/bfcl_v4_multi_turn/partial/coverage"] == pytest.approx(4 / 800)


def test_missing_category_is_labeled_partial():
    result = compute_bfcl_v4_multi_turn_metrics(
        ["bfcl_v4/multi_turn_base"],
        {"acc": [1.0]},
    )
    assert result["val-aux/bfcl_v4_multi_turn/partial/accuracy"] == 1.0
    assert result["val-aux/bfcl_v4_multi_turn/partial/coverage"] == pytest.approx(1 / 800)


def test_non_bfcl_validation_is_ignored():
    assert compute_bfcl_v4_multi_turn_metrics(["gsm8k"], {"acc": [1.0]}) == {}
