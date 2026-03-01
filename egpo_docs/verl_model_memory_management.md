# verl 中的模型管理和内存优化

本文档详细解释 verl 框架中的模型加载、内存管理和参数同步机制。

---

## 模型部署框架

### 训练框架（Actor 和 Ref）

- **FSDP（Fully Sharded Data Parallel）**：PyTorch 的完全分片数据并行
- **Megatron-LM**：NVIDIA 的大规模模型训练框架

这些框架用于：
- Actor 模型的训练（`update_policy`）
- Ref 模型的推理（`compute_log_prob`）

### 推理框架（Rollout）

- **vLLM**：高性能 LLM 推理和服务框架
- **SGLang**：结构化生成语言框架
- **TGI（Text Generation Inference）**：HuggingFace 的推理服务框架

这些框架用于：
- Rollout 模型的序列生成（`generate_sequences`）

---

## ActorRolloutRef Worker 的内存管理

### 设计理念

`ActorRolloutRefWorker` 是一个混合 Worker，可以在同一个 Worker 中同时包含 Actor、Rollout 和 Ref 三个组件。通过模型 offloading 在训练和生成阶段之间切换，优化显存使用。

### 生成阶段（Rollout）

1. **Actor 模型 offload 到 CPU**：
   ```python
   self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
   ```
   - 只 offload 模型参数，保留 optimizer 和梯度在 GPU（如果需要）

2. **Rollout 模型在 GPU 上生成**：
   - 从 Actor 获取最新权重
   - 通过 `update_weights()` 同步到 Rollout 模型
   - 在 GPU 上执行序列生成

3. **显存清理**：
   ```python
   aggressive_empty_cache(force_sync=True)
   ```

### 训练阶段（Trainer）

1. **Rollout 模型 offload（可选）**：
   - 如果配置了 `free_cache_engine`，Rollout 模型会被 offload
   - 否则保留在 GPU（取决于配置）

2. **Actor 模型在 GPU 上训练**：
   - Actor 模型从 CPU 加载回 GPU
   - 执行 policy update

### Ref 模型的特殊处理

- **LoRA 优化**：
  - 如果使用 LoRA，Ref 模型只加载 LoRA 权重
  - 与 Actor 共享基础模型参数，大幅减少显存占用
  - 代码位置：`verl/workers/engine_workers.py` 的 `ActorRolloutRefWorker`

---

## 参数同步机制

### Rollout 模型权重更新

Rollout 模型的参数通过 `update_weights()` 方法从 Actor 同步，**不需要重新部署**。

#### vLLM 的权重更新流程

1. **获取 Actor 权重**：
   ```python
   per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param()
   ```

2. **更新 Rollout 权重**：
   ```python
   await self.rollout.update_weights(per_tensor_param, peft_config=peft_config)
   ```

3. **vLLM 内部处理**：
   - **LoRA 模式**：如果使用 LoRA，只更新 LoRA 权重
   - **FP8 量化**：如果启用 FP8，先将权重转换为 FP8 格式再加载
   - **标准模式**：直接调用 `model.load_weights(weights)`

#### 代码位置

- `verl/workers/rollout/vllm_rollout/vllm_rollout.py` 的 `update_weights` 方法（约 234-270 行）

---

## FP8 量化

### 什么是 FP8？

FP8（8-bit Floating Point）是一种量化技术，使用 8 位浮点数表示模型权重，可以：
- **减少显存占用**：模型大小减少约 50%
- **提高推理速度**：更快的矩阵运算
- **可能影响精度**：需要 TIS（Truncated Importance Sampling）等技术来纠正精度损失

### 在 verl 中的使用

- **Rollout 模型**：可以启用 FP8 量化以提高推理性能
- **权重更新**：vLLM 的 `update_weights` 会自动检测 FP8 并转换权重格式
- **代码位置**：`verl/workers/rollout/vllm_rollout/vllm_rollout.py` 的 `update_weights` 方法

---

## GPU 资源分配

### ResourcePoolManager

`ResourcePoolManager` 管理 GPU 资源池，允许灵活分配 GPU 到不同的 Worker 组。

### 配置示例

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 8

# 可以为不同角色分配不同的 GPU
actor_rollout_ref:
  # Actor/Rollout/Ref 使用 global_pool 的 GPU

reward_model:
  enable_resource_pool: true
  n_gpus_per_node: 2
  nnodes: 1
  # Reward Model 使用独立的 reward_pool GPU
```

### 优势

- **隔离资源**：不同角色可以使用不同的 GPU，避免资源竞争
- **灵活配置**：可以根据需要为不同角色分配不同数量的 GPU
- **代码位置**：`verl/trainer/main_ppo.py` 的 `init_resource_pool_mgr` 方法

---

## 内存优化最佳实践

### 1. 使用 ActorRolloutRef Worker

- 在同一个 Worker 中管理 Actor、Rollout 和 Ref，通过 offloading 优化显存

### 2. 使用 LoRA 共享基础模型

- Ref 模型使用 LoRA，与 Actor 共享基础模型参数

### 3. 启用 FP8 量化（Rollout）

- 如果对精度要求不高，可以为 Rollout 模型启用 FP8 量化

### 4. 合理配置资源池

- 为不同角色分配独立的 GPU 资源池，避免资源竞争

### 5. 调整 offloading 策略

- 根据显存情况调整 `free_cache_engine` 等配置

---

## 相关文档

- [verl 核心概念](./verl_core_concepts.md)
- [verl 中的并行策略](./verl_parallelism_strategies.md)
- [从 main_ppo 到训练结束的完整流程](./main_ppo_training_flow.md)

