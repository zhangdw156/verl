# verl 中的并行策略：TP、DP、PP

本文档详细解释 verl 框架中使用的三种并行策略：Tensor Parallelism (TP)、Data Parallelism (DP) 和 Pipeline Parallelism (PP)。

---

## 概述

在分布式训练和推理中，有三种主要的并行策略来将模型和计算分布到多个 GPU 上：

1. **TP (Tensor Parallelism)**：张量并行
2. **DP (Data Parallelism)**：数据并行
3. **PP (Pipeline Parallelism)**：流水线并行

---

## TP (Tensor Parallelism) - 张量并行

### 原理

**TP 将单个层的权重矩阵分割到多个 GPU 上**，每个 GPU 只存储和计算矩阵的一部分。

### 示例

假设有一个线性层：`Y = X @ W`，其中 `W` 的形状是 `[hidden_size, hidden_size]`。

**不使用 TP**（单 GPU）：
```
GPU 0: W [4096, 4096] → 计算 Y = X @ W
```

**使用 TP（2 个 GPU）**：
```
GPU 0: W_left [4096, 2048]  → 计算 Y_left = X @ W_left
GPU 1: W_right [4096, 2048] → 计算 Y_right = X @ W_right
然后通过 AllReduce 合并：Y = [Y_left, Y_right]
```

### 特点

- **通信开销大**：每个层都需要 AllReduce 通信
- **适合大模型**：当单个 GPU 无法容纳一个层的权重时使用
- **延迟低**：每个 GPU 处理完整的 batch，延迟相对较低

### 在 verl 中的使用

- **训练（Actor/Ref）**：
  - 通过 `tensor_model_parallel_size` 配置
  - 代码位置：`verl/workers/config/engine.yaml` 的 `McoreEngineConfig`

- **推理（Rollout）**：
  - vLLM 支持 TP
  - 通过 `tensor_model_parallel_size` 配置
  - 代码位置：`verl/trainer/config/rollout/rollout.yaml`

### 配置示例

```yaml
# 训练配置
actor_rollout_ref:
  actor:
    engine:
      tensor_model_parallel_size: 2  # 2 个 GPU 做 TP

# 推理配置
rollout:
  tensor_model_parallel_size: 2  # 2 个 GPU 做 TP
```

---

## DP (Data Parallelism) - 数据并行

### 原理

**DP 将完整的模型复制到每个 GPU 上，每个 GPU 处理不同的数据批次**。

### 示例

假设有 4 个 GPU，batch size 为 32。

**使用 DP（4 个 GPU）**：
```
GPU 0: 模型副本 0 → 处理 batch[0:8]
GPU 1: 模型副本 1 → 处理 batch[8:16]
GPU 2: 模型副本 2 → 处理 batch[16:24]
GPU 3: 模型副本 3 → 处理 batch[24:32]
然后通过 AllReduce 同步梯度
```

### 特点

- **通信开销小**：只在反向传播后同步梯度
- **适合中小模型**：模型可以完全放入单个 GPU
- **吞吐量高**：多个 GPU 并行处理不同的数据

### 在 verl 中的使用

- **FSDP（Fully Sharded Data Parallel）**：
  - PyTorch 的完全分片数据并行
  - 通过 `fsdp_size` 配置
  - 代码位置：`verl/workers/config/engine.yaml` 的 `FSDPEngineConfig`

### 配置示例

```yaml
actor_rollout_ref:
  actor:
    engine:
      fsdp_size: 4  # 4 个 GPU 做 FSDP
```

---

## PP (Pipeline Parallelism) - 流水线并行

### 原理

**PP 将模型的不同层分布到不同的 GPU 上，形成一个流水线**。

### 示例

假设模型有 4 层，有 2 个 GPU。

**使用 PP（2 个 GPU）**：
```
GPU 0: Layer 0, Layer 1
GPU 1: Layer 2, Layer 3

处理 batch[0]:
  Step 1: GPU 0 处理 Layer 0, Layer 1 → 输出传给 GPU 1
  Step 2: GPU 1 处理 Layer 2, Layer 3 → 完成

处理 batch[1]:
  Step 1: GPU 0 处理 Layer 0, Layer 1（同时 GPU 1 处理 batch[0] 的 Layer 2, 3）
  Step 2: GPU 1 处理 Layer 2, Layer 3
```

### 特点

- **通信开销中等**：只在层之间传递激活值
- **适合超大模型**：当模型太大无法放入单个 GPU 时使用
- **需要流水线调度**：需要精心设计调度策略以充分利用 GPU

### 在 verl 中的使用

- **Megatron-LM**：
  - 支持 PP
  - 通过 `pipeline_model_parallel_size` 配置
  - 代码位置：`verl/workers/config/engine.yaml` 的 `McoreEngineConfig`

### 配置示例

```yaml
actor_rollout_ref:
  actor:
    engine:
      pipeline_model_parallel_size: 2  # 2 个 GPU 做 PP
```

---

## 组合使用

### TP + PP

当模型非常大时，可以同时使用 TP 和 PP：

```yaml
actor_rollout_ref:
  actor:
    engine:
      tensor_model_parallel_size: 2    # 每层用 2 个 GPU 做 TP
      pipeline_model_parallel_size: 4   # 模型分成 4 段做 PP
      # 总共需要 2 * 4 = 8 个 GPU
```

### DP + TP

也可以组合使用 DP 和 TP：

```yaml
actor_rollout_ref:
  actor:
    engine:
      tensor_model_parallel_size: 2  # 2 个 GPU 做 TP
      # 如果有 8 个 GPU，可以形成 4 个 TP 组，每组 2 个 GPU
```

---

## 训练 vs 推理

### 训练阶段

- **主要使用 TP 和 PP**：
  - TP：分割大层
  - PP：分割模型层
  - FSDP：数据并行 + 参数分片

### 推理阶段

- **主要使用 TP**：
  - vLLM 等推理框架主要支持 TP
  - PP 在推理中较少使用（因为延迟问题）

---

## 选择策略

### 模型大小

- **小模型（< 7B）**：使用 DP（FSDP）
- **中等模型（7B - 70B）**：使用 TP
- **大模型（> 70B）**：使用 TP + PP

### 硬件配置

- **单节点多 GPU**：优先使用 TP
- **多节点**：可以使用 PP 跨节点分布

### 性能考虑

- **延迟敏感**：优先使用 TP（延迟低）
- **吞吐量优先**：可以使用 DP（吞吐量高）

---

## 配置示例

### 单节点 8 GPU，70B 模型

```yaml
actor_rollout_ref:
  actor:
    engine:
      tensor_model_parallel_size: 8  # 8 个 GPU 做 TP
```

### 多节点，超大模型

```yaml
actor_rollout_ref:
  actor:
    engine:
      tensor_model_parallel_size: 4      # 每个节点 4 个 GPU 做 TP
      pipeline_model_parallel_size: 4    # 4 个节点做 PP
      # 总共需要 4 节点 * 4 GPU = 16 个 GPU
```

---

## 相关文档

- [verl 核心概念](./verl_core_concepts.md)
- [verl 中的模型管理和内存优化](./verl_model_memory_management.md)
- [从 main_ppo 到训练结束的完整流程](./main_ppo_training_flow.md)

