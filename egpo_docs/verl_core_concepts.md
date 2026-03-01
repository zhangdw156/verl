# verl 核心概念详解

本文档详细解释 verl 框架中的核心概念：Worker、Actor、Critic、Rollout 等，帮助理解框架的架构设计。

---

## Worker（工作节点）

### 定义

**Worker** 是 verl 框架中的基本计算单元，代表一个在 GPU 上运行的分布式进程，负责执行具体的计算任务。

### 特点

- **运行在 GPU 上**：每个 Worker 通常占用一个或多个 GPU
- **通过 RPC 通信**：Worker 之间通过 Ray 的 RPC 机制进行通信
- **可执行多种角色**：一个 Worker 可以同时承担多个角色（如 Actor、Rollout、Ref）

### 代码位置

- 基类：`verl/single_controller/base/worker.py`
- Worker 组管理：`verl/single_controller/base/worker_group.py`

---

## Actor（策略网络）

### 定义

**Actor** 是 PPO 算法中的策略网络（Policy Network），负责：

1. **计算 log 概率**：`compute_log_prob(batch)` - 计算给定序列的 log 概率
2. **更新策略**：`update_policy(batch)` - 根据 advantage 和 old_log_prob 计算 policy loss，反向传播更新参数

### 关键方法

- `compute_log_prob(batch, calculate_entropy=False)`: 
  - 输入：包含 `input_ids`、`attention_mask` 等的 batch
  - 输出：`old_log_probs` 和可选的 `entropys`（token 级熵）
  - 用途：在训练前计算旧策略的 log 概率，用于重要性采样

- `update_policy(batch)`:
  - 输入：包含 `advantages`、`old_log_probs`、`ref_log_probs` 等的 batch
  - 输出：policy loss 和更新后的参数
  - 用途：执行 PPO 的 policy update

### 部署框架

- **训练后端**：FSDP（Fully Sharded Data Parallel）或 Megatron-LM
- **模型加载**：通过 HuggingFace Transformers（如 `AutoModelForCausalLM`）加载

### 代码位置

- 基类：`verl/workers/actor/base.py`
- 实现：`verl/workers/actor/dp_actor.py`
- Worker 封装：`verl/workers/engine_workers.py` 中的 `TrainingWorker`

---

## Critic（价值函数网络）

### 定义

**Critic** 是 PPO 算法中的价值函数网络（Value Network），负责：

1. **计算价值**：`compute_values(batch)` - 估计状态的价值
2. **更新价值函数**：`update_critic(batch)` - 根据 returns 和 values 更新价值网络

### 关键方法

- `compute_values(batch)`:
  - 输入：包含 `input_ids`、`attention_mask` 等的 batch
  - 输出：`values`（每个 token 位置的价值估计）
  - 用途：用于 GAE（Generalized Advantage Estimation）等需要价值函数的 advantage 估计方法

- `update_critic(batch)`:
  - 输入：包含 `returns`、`values` 等的 batch
  - 输出：critic loss 和更新后的参数
  - 用途：更新价值函数，使其更好地估计回报

### 何时需要 Critic？

- **GAE（Generalized Advantage Estimation）**：需要 Critic 来计算 advantage
- **GRPO/EGPO**：**不需要 Critic**，它们是 critic-less 的算法

### 代码位置

- 基类：`verl/workers/critic/base.py`
- Worker 封装：`verl/workers/engine_workers.py` 中的 `TrainingWorker`

---

## Rollout（行为策略）

### 定义

**Rollout** 是用于数据收集的行为策略（Behavior Policy），负责：

1. **生成序列**：`generate_sequences(batch)` - 根据当前策略采样生成 response
2. **更新权重**：`update_weights(weights)` - 同步 Actor 的权重到 Rollout 模型

### 关键方法

- `generate_sequences(batch)`:
  - 输入：包含 `input_ids`、`attention_mask` 等的 batch（prompt）
  - 输出：包含 `responses`、`input_ids`、`attention_mask` 等的 DataProto
  - 用途：对每个 prompt 采样生成多条 response（例如 4 条），用于后续的 reward 计算和训练

- `update_weights(weights)`:
  - 输入：从 Actor 模型获取的权重生成器
  - 输出：无
  - 用途：将 Actor 的更新后的权重同步到 Rollout 模型，确保 Rollout 使用最新的策略进行生成

### 部署框架

- **推理后端**：vLLM、SGLang、TGI（Text Generation Inference）
- **支持量化**：支持 FP8 量化以提高推理速度和减少显存占用

### 代码位置

- 基类：`verl/workers/rollout/base.py`
- vLLM 实现：`verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- HuggingFace 实现：`verl/workers/rollout/hf_rollout.py`
- Naive 实现：`verl/workers/rollout/naive/naive_rollout.py`

---

## Ref（参考策略）

### 定义

**Ref** 是参考策略（Reference Policy），用于计算 KL 散度，防止策略偏离初始策略太远。

### 用途

1. **KL 散度计算**：
   - 在 reward 中：`use_kl_in_reward: true` - 在 reward 中减去 KL 惩罚
   - 在 loss 中：`use_kl_loss: true` - 在 policy loss 中加上 KL 项

2. **GRPO/EGPO 算法**：
   - GRPO 和 EGPO 虽然不需要 Critic，但通常需要 Ref 模型来计算 KL 散度进行正则化

### 特点

- **通常与 Actor 共享基础模型**：使用 LoRA（Low-Rank Adaptation）等技术，Ref 模型只加载 LoRA 权重，与 Actor 共享基础模型参数，减少显存占用
- **参数不更新**：Ref 模型的参数在训练过程中保持不变，代表初始策略

### 代码位置

- Worker 封装：`verl/workers/engine_workers.py` 中的 `TrainingWorker`

---

## ActorRolloutRef（混合 Worker）

### 定义

**ActorRolloutRef** 是一个混合 Worker，可以在同一个 Worker 中同时包含 Actor、Rollout 和 Ref 三个组件。

### 设计目的

1. **代码复用**：共享模型加载和初始化逻辑
2. **快速权重传输**：Actor 和 Rollout 在同一个 Worker 中，权重同步更快
3. **显存优化**：通过模型 offloading 在训练和生成阶段之间切换

### 内存管理

- **生成阶段（Rollout）**：
  - Actor 模型 offload 到 CPU
  - Rollout 模型在 GPU 上生成序列
  
- **训练阶段（Trainer）**：
  - Rollout 模型 offload（可选，取决于配置）
  - Actor 模型在 GPU 上训练

- **Ref 模型**：
  - 如果使用 LoRA，只加载 LoRA 权重，显存占用小
  - 如果不使用 LoRA，可能需要 offload 管理

### 代码位置

- 实现：`verl/workers/engine_workers.py` 中的 `ActorRolloutRefWorker`

---

## WorkerGroup（工作节点组）

### 定义

**WorkerGroup** 管理一组相同角色的 Worker，作为控制器（Controller）的代理。

### 功能

- **批量操作**：对组内所有 Worker 执行相同的操作（如 `generate_sequences`、`update_policy`）
- **负载均衡**：在多个 Worker 之间分配任务
- **故障处理**：管理 Worker 的生命周期和故障恢复

### 代码位置

- 基类：`verl/single_controller/base/worker_group.py`

---

## 总结

| 组件 | 主要职责 | 部署框架 | 是否需要（GRPO/EGPO） |
|------|---------|---------|---------------------|
| **Worker** | 基本计算单元 | Ray | 必需 |
| **Actor** | 策略网络，计算 log_prob 和更新策略 | FSDP/Megatron | 必需 |
| **Critic** | 价值函数网络 | FSDP/Megatron | **不需要**（GRPO/EGPO 是 critic-less） |
| **Rollout** | 行为策略，生成序列 | vLLM/SGLang/TGI | 必需 |
| **Ref** | 参考策略，计算 KL 散度 | FSDP/Megatron | 通常需要（用于 KL 正则化） |
| **ActorRolloutRef** | 混合 Worker，包含 Actor/Rollout/Ref | 混合 | 可选（用于优化资源使用） |

---

## 相关文档

- [GRPO 和 EGPO 算法详解](./EGPO_step_by_step_guide.md)
- [verl 中的模型管理和内存优化](./verl_model_memory_management.md)
- [从 main_ppo 到训练结束的完整流程](./main_ppo_training_flow.md)

