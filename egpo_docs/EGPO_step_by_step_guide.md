# 一步步理解 verl 里的 GRPO 和 EGPO

## 第一步：一轮 PPO 训练在做什么（大流程）

在 `verl/trainer/ppo/ray_trainer.py` 的 `fit()` 里，每一轮大致是：

1. **Rollout**：同一个 prompt 采样多条 response（例如 4 条），batch 里每条样本有一个 **uid**（同一 prompt 的多条共享同一个 uid）。
2. **Reward**：对每条 response 算分 → 得到 `token_level_rewards`（shape: `bs × response_length`），通常每条约等于一个标量分数（对 token 维求和）。
3. **old_log_prob**：用当前 actor 重新算这批 response 的 log 概率（以及可选地 **熵**），得到 `old_log_probs` 等，和 batch 合并。
4. **Advantage**：用 `token_level_rewards`、`response_mask`、`uid` 等算 advantage，写回 `batch.batch["advantages"]`。
5. **Update actor**：用 `advantages` 和 `old_log_probs` 等算 policy loss，反向传播更新 actor。

你要关心的只有两处：**advantage 怎么算**（GRPO/EGPO 的差别在这里），以及 **算 advantage 时用到的数据从哪来**（比如熵）。

---

## 第二步：advantage 是在哪里算的？

入口是**同一个函数**：

- 文件：`verl/trainer/ppo/ray_trainer.py`
- 函数：`compute_advantage(data, adv_estimator, ...)`（约 186 行）

里面根据 `adv_estimator` 分支：

- `AdvantageEstimator.GAE` → 用 GAE 算 advantage（需要 value）
- `AdvantageEstimator.GRPO` → 用 GRPO 算
- `AdvantageEstimator.EGPO` → 用 EGPO 算
- 其他 → 用注册的别的估计器

所以：**GRPO 和 EGPO 都是"在 compute_advantage 里选不同分支，用不同算法算 advantage"**。

---

## 第三步：GRPO 的 advantage 怎么算？

GRPO 的实现在：

- 文件：`verl/trainer/ppo/core_algos.py`
- 函数：`compute_grpo_outcome_advantage`（约 268 行）

**输入：**

- `token_level_rewards`：`(bs, response_length)`，每条 response 的 token 奖励
- `response_mask`：哪些位置是有效 response token
- `index`：即 **uid**，`(bs,)`，用来分组（同一 prompt 的多条 response 的 uid 相同）

**逻辑（简化）：**

1. 每条 response 一个分数：`scores[i] = token_level_rewards[i].sum()`（标量）。
2. 按 **uid（index）分组**，同一组内算均值和标准差：  
   `mean_g`、`std_g`。
3. 组内标准化：  
   `A_i = (scores[i] - mean_g) / (std_g + eps)`  
   （若 `norm_adv_by_std_in_grpo=False` 则只减均值不除 std）。
4. 把标量 `A_i` 扩成 token 维：  
   `advantages[i, :] = A_i * response_mask[i, :]`。

所以 **GRPO 的 advantage 只依赖：每条 response 的总分、uid 分组、response_mask**，不依赖熵。

---

## 第四步：EGPO 在 GRPO 基础上多做了什么？

EGPO 的实现在同一文件：

- 文件：`verl/trainer/ppo/core_algos.py`
- 函数：`compute_egpo_outcome_advantage`（约 402 行）

**和 GRPO 的对比：**

- 输入**多了一个**：`token_level_entropy`，shape `(bs, response_length)`，即每个 token 的熵。
- 输入**多了一个**：`responses`，shape `(bs, response_length)`，用于创建 CoT mask。
- 前面几步和 GRPO **完全一样**：先按 `token_level_rewards` 和 uid 算出 GRPO 的标量 `A_i`。
- 多出来的步骤：
  1. 从 `responses` 创建 **CoT mask**：使用 Qwen3 的 `<think>` (token ID 151667) 和 `</think>` (token ID 151668) 标记，识别 CoT 部分的 token。
  2. 对每条 response 在 **CoT mask** 下求**平均熵**：  
     `H_i = masked_mean(token_level_entropy[i], cot_entropy_mask[i])`，得到标量 `H_i`。
  3. 裁剪熵，不改变 `A_i` 的符号：  
     `bound = |A_i| / alpha`，  
     `entropy_bonus = clip(H_i, -bound, bound)`。
  4. EGPO 的 advantage：  
     `A_egpo_i = A_i + lambda * entropy_bonus`，  
     再像 GRPO 一样 broadcast 成 token 维乘上 `response_mask`。

所以：**EGPO = 先用 GRPO 的公式算 A_i，再加上一个"裁剪过的 CoT 熵项"，再 broadcast；policy loss 仍然用现有的 PPO/GRPO 那套，没有改 loss 公式。**

**重要**：EGPO 只使用 **Chain-of-Thought (CoT) 部分的熵**，而不是整个 response 的熵，这符合论文的规范。

---

## 第五步：EGPO 的熵从哪里来？

EGPO 需要 **token 级熵** 在 `compute_advantage` 时已经在 `data.batch["entropys"]` 里。整条数据流是：

1. **谁算熵？**  
   Actor 的 `compute_log_prob`。  
   文件：`verl/workers/actor/dp_actor.py`，函数 `compute_log_prob`（约 425 行）。  
   当 `calculate_entropy=True` 或 `data.meta_info["calculate_entropy"]=True` 时，会多算一份 `entropys` 并放进返回的 batch。

2. **谁在 EGPO 时请求熵？**  
   Trainer 在 recompute old_log_prob 之前，如果是 EGPO，就设：  
   `batch.meta_info["calculate_entropy"] = True`。  
   文件：`verl/trainer/ppo/ray_trainer.py`，约 1412–1414 行。

3. **算完 old_log_prob 之后，entropys 要不要保留？**  
   非 EGPO 时，会把 `old_log_prob.batch` 里的 `entropys` 删掉（避免占显存、避免误用）。  
   EGPO 时**不删**，这样 `batch.union(old_log_prob)` 之后，合并的 batch 里还有 `entropys`，供后面 `compute_advantage` 用。  
   文件：同上，约 1429–1431 行。

4. **responses 字段从哪里来？**  
   `responses` 字段在 rollout 生成后就已经存在于 `data.batch` 中。各种 rollout 实现（naive_rollout、hf_rollout、agent_loop 等）都会返回包含 `responses` 的 DataProto。

5. **用完熵之后？**  
   在 `compute_advantage` 的 EGPO 分支里，算完 advantage 后会对 `data.batch` 做 `pop("entropys")`，避免后面步骤再误用。  
   文件：`verl/trainer/ppo/ray_trainer.py`，约 253–266 行。

你只要记住：**EGPO 分支需要 `data.batch["entropys"]` 和 `data.batch["responses"]`，所以 trainer 在 EGPO 时让 actor 算熵、并且不 pop 掉 entropys，直到 advantage 算完再 pop。**

---

## 第六步：配置上怎么选 GRPO / EGPO？

在 algorithm 配置里（例如 `verl/trainer/config/algorithm.py` 里的 `AlgoConfig`）：

- `adv_estimator: "grpo"` → 用 GRPO 算 advantage（不涉及熵）。
- `adv_estimator: "egpo"` → 用 EGPO 算 advantage（需要熵；trainer 会自动设 `calculate_entropy` 并保留 `entropys`）。
- EGPO 额外参数：
  - `egpo_lambda`（默认 0.4）：熵项的权重
  - `egpo_alpha`（默认 2.0）：裁剪阈值，保证熵项不会反转 advantage 的符号
  - `cot_start_token_id`（默认 151667）：CoT 开始 token ID（Qwen3 的 `<think>`）
  - `cot_end_token_id`（默认 151668）：CoT 结束 token ID（Qwen3 的 `</think>`）

在 `compute_egpo_outcome_advantage` 里会从 config 读取这些参数。

---

## 小结（对照代码看）

| 你想搞清的 | 位置 |
|-----------|------|
| 一轮训练里 reward → advantage → update 的先后顺序 | `ray_trainer.py` 的 `fit()`，约 1386 reward → 1410 old_log_prob → 1460 token_level_rewards → 1499 compute_advantage → 1514 update_actor |
| advantage 用哪种算法算 | `ray_trainer.py` 的 `compute_advantage()`，按 `adv_estimator` 分支 |
| GRPO 的公式（按 uid 分组、标准化、broadcast） | `core_algos.py` 的 `compute_grpo_outcome_advantage()` |
| EGPO 的公式（GRPO + 裁剪 CoT 熵项） | `core_algos.py` 的 `compute_egpo_outcome_advantage()` |
| CoT mask 的创建 | `core_algos.py` 的 `_create_cot_mask_from_redacted_reasoning()` |
| 熵是谁算的、什么时候保留、什么时候 pop | `dp_actor.py` 的 `compute_log_prob`；`ray_trainer.py` 里设 `calculate_entropy`、不 pop entropys（EGPO）、算完 advantage 后 pop |

按上面六步顺着走一遍，再对着这些行号看代码，就能把 GRPO 和 EGPO 在 verl 里是怎么实现的串起来。

