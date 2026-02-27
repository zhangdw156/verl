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
"""
Single-Criteria binary reward for function calling (EGPO-style).
Reward = 1.0 only when both format and correctness pass; otherwise 0.0.
Reference: Reasoning through Exploration (Hao et al., 2025, arXiv:2508.05118).
"""
import re
from typing import Any, Optional


def _normalize_for_compare(s: str) -> str:
    """Normalize string for comparison: strip, collapse whitespace, optional lower."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _check_format(response_str: str, require_thinking: bool = True) -> bool:
    """Check that response has CoT (e.g. </think>) and tool-call-like content."""
    if require_thinking:
        if "</think>" not in response_str or "<think>" not in response_str:
            return False
    # Tool-call indicator: [func(...)] or {"name": or "function" or similar
    tool_pattern = r"\[[\w_]+\s*\([^)]*\)\]|\{\s*[\"']name[\"']\s*:|[\"']function[\"']\s*:"
    if not re.search(tool_pattern, response_str, re.IGNORECASE):
        return False
    return True


def _extract_answer_part(response_str: str) -> str:
    """Extract the answer / tool-call part after </think> (or full string if no tag)."""
    if "</think>" in response_str:
        # Take content after last </think>
        parts = response_str.split("</think>")
        return parts[-1].strip() if parts else response_str.strip()
    return response_str.strip()


def compute_score(
    solution_str: str,
    ground_truth: str,
    check_format: bool = True,
    require_thinking: bool = True,
    normalize_ground_truth: bool = True,
    **kwargs: Any,
) -> float:
    """Single-Criteria binary reward for function calling (EGPO).

    Returns 1.0 only when:
      - (Optional) Format passes: response contains <think>...</think> and tool-call-like content.
      - Correctness: extracted answer part matches ground_truth after normalization.

    Otherwise returns 0.0.

    Args:
        solution_str: Model response (may include <think>...</think> and tool call).
        ground_truth: Reference answer (e.g. reference tool call string or JSON).
        check_format: If True, require format check (CoT + tool call pattern).
        require_thinking: If True, require <think> and </think> tags when check_format is True.
        normalize_ground_truth: If True, normalize ground_truth before comparison.
        **kwargs: Ignored (for API compatibility with default_compute_score).

    Returns:
        1.0 if both format and correctness pass, else 0.0.
    """
    if not solution_str or not str(ground_truth).strip():
        return 0.0

    if check_format and not _check_format(solution_str, require_thinking=require_thinking):
        return 0.0

    pred = _normalize_for_compare(_extract_answer_part(solution_str))
    ref = _normalize_for_compare(str(ground_truth)) if normalize_ground_truth else str(ground_truth).strip()

    return 1.0 if pred == ref else 0.0
