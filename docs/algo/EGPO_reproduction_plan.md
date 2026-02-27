# EGPO 复现方案：在 verl 中实现论文算法

## 论文与算法概要

**论文**: *Reasoning through Exploration: A Reinforcement Learning Framework for Robust Function Calling* (Hao et al., 2025, arXiv:2508.05118)

**核心思想**: EGPO 基于 GRPO，通过 **熵增强的 advantage** 把模型 Chain-of-Thought (CoT) 的熵纳入策略梯度，促进多样推理路径的探索；熵项用 **裁剪** 限制，保证不改变原始 advantage 的符号，再配合 **严格二元 reward**，用于 function calling / tool use 的鲁棒训练。

### 关键公式

- **GRPO advantage**（同 verl 现有实现）:
  $$A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_S\})}{\text{std}(\{r_1,\dots,r_S\})}$$

- **EGPO 熵增强 advantage**:
  $$A_i^{\text{EGPO}} = A_i + \lambda \cdot \text{clip}\left(H_i,\ -\frac{|A_i|}{\alpha},\ \frac{|A_i|}{\alpha}\right)$$
  - \(H_i\): 第 \(i\) 条 rollout 的 **CoT 熵**（可先简化为整条 response 在 mask 下的平均 token 熵）
  - \(\lambda > 0\): 熵权重（论文 \(\lambda=0.4\)）
  - \(\alpha > 1\): 裁剪阈值（论文 \(\alpha=2\)），保证熵项不会反转 \(A_i\) 的符号

### 论文设定摘要

- **Reward**: Single-Criteria，严格二元——仅当格式与语义都正确时给 reward。
- **数据**: xlam-function-calling-60k、Open-Agentic-tool-use 等；response 含 CoT（如 `<thinking>...</thinking>`）+ function call。
- **实现**: 论文注明使用 **Verl** 框架；5 epochs，lr=1e-6，temperature=0.7，KL coeff=0.001，max response 8192，\(\lambda=0.4\)，\(\alpha=2\)。

---

## 在 verl 中的复现思路

### 1. 算法层面（与现有组件的对应关系）

| 论文概念 | verl 中对应 |
|---------|-------------|
| GRPO | `AdvantageEstimator.GRPO`，`compute_grpo_outcome_advantage` |
| Group / 同 prompt 多 rollout | `uid`（`data.non_tensor_batch["uid"]`），同 uid 为一组 |
| Outcome reward \(r_i\) | `token_level_rewards` 按 token 求和得到的 per-response 标量 |
| CoT 熵 \(H_i\) | 当前 policy 在 **response 部分**的 token 级熵，再对有效 token 取平均得到 per-sample 标量 |

### 2. 实现要点

#### 2.1 新增 EGPO Advantage Estimator

- **位置**: `verl/trainer/ppo/core_algos.py`
- **步骤**:
  1. 用现有 GRPO 逻辑算出 \(A_i\)（per-sample 标量，再 broadcast 到 token 维）。
  2. 从 `data.batch["entropys"]` 得到 token 级熵 `(bs, response_length)`，在 `response_mask` 下对每个 sample 求平均，得到 \(H_i\)，shape `(bs,)`。
  3. 对每个 sample：  
     `A_egpo_i = A_i + lambda * clip(H_i, -|A_i|/alpha, |A_i|/alpha)`  
     再按 token 维 broadcast 并乘 `response_mask`。
- **配置**: 在 `AlgoConfig`（或等价 algorithm config）中增加：
  - `egpo_lambda: float = 0.4`
  - `egpo_alpha: float = 2.0`
  - `adv_estimator: str = "egpo"` 时使用上述逻辑。

#### 2.2 保证 batch 中有 `entropys`

- **当前行为**: `_compute_old_log_prob` 已可返回 `entropys`；在 fit 里为了指标会做一次聚合，然后 `old_log_prob.batch.pop("entropys")`，再 `batch.union(old_log_prob)`，导致合并后的 batch 没有 `entropys`。
- **修改**: 当 `adv_estimator == "egpo"` 时 **不要** 从 `old_log_prob` 里 pop 掉 `entropys`，这样 `batch.union(old_log_prob)` 后 `batch.batch["entropys"]` 存在，供 `compute_advantage` 使用。
- **Legacy actor 路径**: 若通过 `compute_log_prob(batch)` 且未显式传 `calculate_entropy`，需要在该路径下当 `adv_estimator == "egpo"` 时也请求熵（例如在 trainer 里根据 `self.config.algorithm.adv_estimator == "egpo"` 设置 `calculate_entropy=True` 或等价逻辑），保证 legacy 和 non-legacy 都能在 EGPO 下拿到 `entropys`。

#### 2.3 Reward 与数据

- **Single-Criteria 二元 reward**: 已实现于 `verl.utils.reward_score.function_calling`：格式（CoT + tool-call 模式）与正确性（与 ground_truth 归一化比较）均通过则 1.0，否则 0.0；数据中 `data_source` 设为 `function_calling` 等即可接入。
- **CoT 区域（可选细化）**: 论文写的是 “entropy of the model’s Chain-of-Thought”。若数据里 CoT 有明确边界（如 `<thinking>...</thinking>`），可后续增加可选的 `cot_mask`，只在 CoT 位置上平均熵得到 \(H_i\)；首版复现可简化为 **整条 response 在 response_mask 下的平均熵**。

### 3. 配置与入口

- **Algorithm**:
  - `adv_estimator: "egpo"`
  - `egpo_lambda: 0.4`, `egpo_alpha: 2.0`
  - 其他沿用 GRPO（如 `norm_adv_by_std_in_grpo` 等按需保留）。
- **Reward**: 使用 custom_reward_function 或现有 reward_manager，实现 Single-Criteria 二元评分。
- **数据**: 使用 BFCL / function calling 类数据集（如 xlam、Open-Agentic-tool-use），prompt 含 tools，response 含 CoT + tool call；与现有 PPO 数据格式一致即可。

### 4. 与现有代码的衔接

- **GRPO**: EGPO 先算 GRPO advantage，再在标量 \(A_i\) 上加上裁剪后的熵项并 broadcast，不改变现有 GRPO 的 group/uid 语义。
- **Actor entropy**: 已有 `calculate_entropy` 与 token 级 `entropys`，仅需在 EGPO 分支保留并传入 `compute_advantage`。
- **Policy loss**: 仍用现有 PPO/GRPO 的 clip objective；论文未在 loss 里再加熵项，仅用熵改 advantage，因此无需改 loss 函数。

---

## 实现清单（简要）

1. **core_algos.py**
   - 增加 `AdvantageEstimator.EGPO`。
   - 实现 `compute_egpo_outcome_advantage(token_level_rewards, response_mask, index, token_level_entropy, egpo_lambda, egpo_alpha, norm_adv_by_std_in_grpo, ...)`：内部先调 GRPO 得 \(A_i\)，再算 \(H_i\) 与 clip，得到 EGPO advantage 与 returns。

2. **algorithm config**
   - 增加 `egpo_lambda: float = 0.4`、`egpo_alpha: float = 2.0`（或从 config 的 get 读取）。

3. **ray_trainer.py（及同逻辑 trainer）**
   - 在 `compute_advantage` 中为 `adv_estimator == "egpo"` 增加分支，传入 `data.batch["entropys"]` 与 egpo 超参。
   - 当 `adv_estimator == "egpo"` 时，**不要** 对 `old_log_prob.batch` 执行 `pop("entropys")`，以便 `entropys` 进入合并后的 batch。
   - Legacy 路径下，当 `adv_estimator == "egpo"` 时确保调用 actor 的 `compute_log_prob` 时带有 `calculate_entropy=True`（或等价配置）。

4. **Reward**
   - 已实现：`verl/utils/reward_score/function_calling.py`，通过 `data_source` 为 `function_calling` / `bfcl` / `xlam-function-calling` / `open_agentic_tool_use` 时由 `default_compute_score` 自动调用。

5. **实验**
   - 用 BFCL / 论文提到的数据集与评估脚本验证 EGPO 相比 GRPO 在 Multi-Turn 与 Single-Turn 上的提升。

按上述步骤即可在 verl 中完整复现论文中的 EGPO 算法，并与现有 GRPO、reward、数据流程兼容。

---

## 已实现改动摘要（verl 代码库）

- **`verl/trainer/ppo/core_algos.py`**: 新增 `AdvantageEstimator.EGPO` 与 `compute_egpo_outcome_advantage()`，先算 GRPO advantage，再按论文公式加上裁剪后的熵项。
- **`verl/trainer/config/algorithm.py`**: `AlgoConfig` 增加 `egpo_lambda`（默认 0.4）、`egpo_alpha`（默认 2.0）。
- **`verl/trainer/ppo/ray_trainer.py`**:  
  - `compute_advantage()` 中增加 EGPO 分支，传入 `entropys` 与 config；  
  - 当 `adv_estimator == EGPO` 时不再从 `old_log_prob` 中 `pop("entropys")`，并在计算完 advantage 后从 `data.batch` 中 `pop("entropys")`；  
  - 在 recompute old_log_prob 前设置 `batch.meta_info["calculate_entropy"] = True`（供 legacy actor 使用）。
- **`verl/workers/actor/dp_actor.py`**: `compute_log_prob()` 中增加对 `data.meta_info.get("calculate_entropy", False)` 的读取，以便 EGPO 在 legacy 路径下也能拿到 token 级熵。

**配置示例**（YAML 或 DictConfig）:

```yaml
algorithm:
  adv_estimator: egpo
  norm_adv_by_std_in_grpo: true
  egpo_lambda: 0.4
  egpo_alpha: 2.0
```

**Reward**: 已实现。数据中 `data_source`（或 reward_fn_key）设为 `function_calling`、`bfcl`、`xlam-function-calling`、`open_agentic_tool_use` 之一即可使用 `verl.utils.reward_score.function_calling.compute_score`（Single-Criteria 二元：格式 + 正确性通过才 1.0，否则 0.0）。
