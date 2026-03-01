# verl 中的 Agent Loop 使用指南

本文档详细解释 verl 框架中的 Agent Loop 功能，包括多轮对话、工具调用和 agentic 强化学习。

---

## 什么是 Agent Loop？

**Agent Loop** 是 verl 框架中用于多轮交互和 agentic 强化学习的通用接口，支持：

1. **多轮对话**：模型可以与用户或环境进行多轮交互
2. **工具调用**：模型可以调用外部工具（如 API、函数等）
3. **环境交互**：模型可以与外部环境交互（如游戏、模拟器等）
4. **多模态支持**：支持图像、视频等多模态输入和输出

---

## Agent Loop 的工作流程

### 基本流程

```
1. 初始化：接收初始 prompt 和工具定义
2. 生成：模型生成 response（可能包含工具调用）
3. 工具执行：如果模型调用了工具，执行工具并获取结果
4. 环境交互（可选）：如果配置了环境，与环境交互
5. 更新历史：将工具结果和环境响应添加到对话历史
6. 继续生成：模型基于更新后的历史继续生成
7. 终止：达到终止条件（最大轮数、用户终止等）
```

### 状态机

Agent Loop 使用状态机管理流程：

- **PENDING**：初始状态，准备生成
- **GENERATING**：正在生成 response
- **PROCESSING_TOOLS**：正在处理工具调用
- **INTERACTING**：正在与环境交互
- **TERMINATED**：终止状态

---

## 配置 Agent Loop

### Rollout 配置

在 rollout 配置中启用 agent loop：

```yaml
rollout:
  name: vllm  # 或 sglang
  # ... 其他配置
```

### Agent Loop 配置

```yaml
agent_loop:
  max_assistant_turns: 5      # 最大 assistant 轮数
  max_user_turns: 10           # 最大 user 轮数
  max_parallel_calls: 3        # 最大并行工具调用数
  response_length: 8192        # 最大 response 长度
  tool_config_file: tools.yaml # 工具配置文件
  interaction_config_file: interaction.yaml  # 环境交互配置文件（可选）
```

### 工具配置

在 `tools.yaml` 中定义工具：

```yaml
tools:
  - name: search_web
    description: "Search the web for information"
    parameters:
      type: object
      properties:
        query:
          type: string
          description: "The search query"
      required:
        - query
```

### 环境交互配置（可选）

在 `interaction.yaml` 中配置环境交互：

```yaml
interaction:
  type: custom  # 自定义交互类型
  # ... 其他配置
```

---

## 数据格式

### 输入数据格式

Agent Loop 需要的数据格式：

```python
{
    "messages": [
        {"role": "user", "content": "请帮我搜索..."}
    ],
    "tools": [...],  # 工具定义（可选）
    "tool_choice": "auto"  # 工具选择策略（可选）
}
```

### 输出数据格式

Agent Loop 返回的 `DataProto` 包含：

```python
{
    "prompts": torch.Tensor,      # [bsz, prompt_length]
    "responses": torch.Tensor,    # [bsz, response_length]
    "response_mask": torch.Tensor, # [bsz, response_length]
    "input_ids": torch.Tensor,    # [bsz, prompt_length + response_length]
    "attention_mask": torch.Tensor, # [bsz, prompt_length + response_length]
    "position_ids": torch.Tensor,  # [bsz, prompt_length + response_length]
    # ... 其他字段
}
```

**注意**：`responses` 字段包含多轮对话的完整内容：
```
responses: |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
```

---

## 工具调用

### 工具定义

工具使用 OpenAI 的函数调用格式定义：

```python
{
    "name": "function_name",
    "description": "Function description",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."}
        },
        "required": ["param1"]
    }
}
```

### 工具执行

当模型生成工具调用时：

1. **解析工具调用**：从 response 中提取工具调用
2. **执行工具**：调用工具函数并获取结果
3. **处理多模态输出**：如果工具返回图像/视频，进行编码处理
4. **更新对话历史**：将工具结果添加到消息历史

### 多模态工具

如果工具返回图像或视频：

```python
# 工具返回图像
tool_response.image = [Image1, Image2, ...]

# Agent Loop 会：
# 1. 将图像添加到 agent_data.image_data
# 2. 在消息中使用结构化格式：{"type": "image"}
# 3. 通过 apply_chat_template 编码到 input_ids
```

---

## 环境交互

### 配置环境

如果配置了 `interaction_config_file`，Agent Loop 会与环境交互：

```python
# 环境交互流程
1. 模型生成 response
2. 调用 interaction.generate_response()
3. 获取环境响应和 reward
4. 将环境响应添加到对话历史
5. 继续生成或终止
```

### 使用场景

- **游戏环境**：模型与游戏环境交互
- **模拟器**：模型与物理模拟器交互
- **API 环境**：模型与外部 API 交互

---

## 与 PPO 训练的集成

### 数据流

1. **生成阶段**：
   - Agent Loop 生成多轮对话和工具调用
   - 返回包含完整 `responses` 的 DataProto

2. **Reward 计算**：
   - 基于完整的对话历史计算 reward
   - 可以基于工具调用结果、环境响应等

3. **训练阶段**：
   - 使用标准的 PPO 流程
   - `responses` 字段用于计算 response_mask 等

### 特殊考虑

- **多轮对话的 reward**：
  - 可以在每轮结束时计算中间 reward
  - 也可以在最终结束时计算总 reward

- **工具调用的处理**：
  - 工具调用的 token 在 `response_mask` 中标记为 0（非 LLM 生成）
  - 只有 LLM 生成的 token 参与训练

---

## 代码位置

### Agent Loop 实现

- 基类：`verl/experimental/agent_loop/agent_loop.py` 的 `AgentLoopWorkerBase`
- Tool Agent Loop：`verl/experimental/agent_loop/tool_agent_loop.py` 的 `ToolAgentLoop`
- 管理器：`verl/experimental/agent_loop/agent_loop.py` 的 `AgentLoopManager`

### Rollout 集成

- vLLM Rollout：`verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- Agent Loop 调用：通过 `AsyncRolloutManager` 调用 `AgentLoopManager`

---

## 使用示例

### 基本配置

```yaml
# config.yaml
rollout:
  name: vllm
  tensor_model_parallel_size: 1
  gpu_memory_utilization: 0.9

agent_loop:
  max_assistant_turns: 5
  max_user_turns: 10
  max_parallel_calls: 3
  response_length: 8192
  tool_config_file: tools.yaml
```

### 数据示例

```python
# 数据集中的一条样本
{
    "messages": [
        {"role": "user", "content": "请帮我搜索 Python 教程"}
    ],
    "tools": [
        {
            "name": "search_web",
            "description": "搜索网页",
            "parameters": {...}
        }
    ]
}
```

---

## 注意事项

1. **Response 长度**：
   - Agent Loop 可能生成很长的 response（包含多轮对话）
   - 需要合理设置 `response_length`

2. **工具调用格式**：
   - 确保模型支持工具调用格式（如 OpenAI 格式）
   - 工具定义需要与模型训练时的格式一致

3. **多模态处理**：
   - 如果使用多模态工具，确保模型支持多模态输入
   - 需要正确配置 processor

4. **Reward 设计**：
   - 多轮对话的 reward 设计需要考虑中间步骤
   - 工具调用的成功/失败可以作为 reward 信号

---

## 相关文档

- [verl 核心概念](./verl_core_concepts.md)
- [GRPO 和 EGPO 算法详解](./EGPO_step_by_step_guide.md)
- [从 main_ppo 到训练结束的完整流程](./main_ppo_training_flow.md)

