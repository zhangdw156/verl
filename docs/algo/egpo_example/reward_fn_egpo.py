"""
EGPO 用严格二元 reward（Hao et al. 2025）：
仅当「格式正确」且「工具调用与 ground_truth 一致」时返回 1.0，否则 0.0。
与 GRPO 用的 reward_fn 逻辑兼容（复用 extract_tool_calls、compare_parsed_content），
仅把得分改为 0/1。
"""
import json
import re
from collections import Counter


def _convert_to_hashable(data):
    if isinstance(data, dict):
        return frozenset((key, _convert_to_hashable(value)) for key, value in data.items())
    if isinstance(data, list):
        return frozenset(_convert_to_hashable(item) for item in data)
    return data


def compare_parsed_content(parsed1, parsed2):
    """比较两个工具调用列表，忽略顺序。"""
    counter1 = Counter([_convert_to_hashable(item) for item in parsed1])
    counter2 = Counter([_convert_to_hashable(item) for item in parsed2])
    return counter1 == counter2


def extract_tool_calls(input_string):
    """从文本中提取 <tool_call> 标签内的 JSON 内容。"""
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, input_string, re.DOTALL)
    result = []
    for match in matches:
        try:
            result.append(json.loads(match))
        except Exception:
            result.append(match)
    return result


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    EGPO 严格二元奖励：格式通过且内容正确才 1.0，否则 0.0。
    与 verl 自定义奖励入口签名一致。
    """
    if not (solution_str and str(ground_truth).strip()):
        return 0.0

    has_tool = "<tool_call>" in solution_str and "</tool_call>" in solution_str
    if not has_tool:
        return 0.0

    try:
        gt_tools = extract_tool_calls(ground_truth)
        pd_tools = extract_tool_calls(solution_str)
        if not pd_tools:
            return 0.0
        if compare_parsed_content(gt_tools, pd_tools):
            return 1.0
    except Exception:
        pass
    return 0.0
