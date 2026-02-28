# 从 main_ppo 到训练结束：完整流程与涉及代码

你通过 `python3 -m verl.trainer.main_ppo --config-path=xxx --config-name=yyy` 启动时，Hydra 会加载指定目录下的 yaml，与默认配置合并后得到 `config`。下面按执行顺序说明每一步做了什么、涉及哪些代码。

---

## 一、入口与配置加载

### 1. 命令行入口

- **你执行的**：`python3 -m verl.trainer.main_ppo --config-path=目录 --config-name=yaml名`
- **实际调用的模块**：`verl/trainer/main_ppo.py`（`python -m` 会执行该包下的 `__main__`，即 `main_ppo.py`）。

### 2. Hydra 装饰器与 main()

- **文件**：`verl/trainer/main_ppo.py`
- **第 36 行**：`@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)`
  - 若不传 `--config-path`，默认在 **相对 main_ppo.py 所在目录** 下找 `config` 目录（即 `verl/trainer/config/`）。
  - 你传了 `--config-path` 和 `--config-name` 时，Hydra 会从你指定的目录加载你指定的 yaml，并与该 yaml 里 `defaults:` 引用的其他 yaml（如 `ppo_trainer.yaml` 里的 algorithm、actor、data、reward 等）合并。
- **第 37–46 行**：`main(config)`  
  - 先 `auto_set_device(config)`（例如 NPU 时改 device）、`migrate_legacy_reward_impl(config)` 做兼容，然后调用 `run_ppo(config)`。

---

## 二、run_ppo(config)：Ray 与 TaskRunner

- **文件**：`verl/trainer/main_ppo.py`，函数 `run_ppo`（约 49–105 行）。

### 1. 初始化 Ray

- **约 58–77 行**：若 Ray 未初始化，则 `ray.init(**ray_init_kwargs)`，用 `config.ray_kwargs` 和默认 runtime_env 起本地或集群。

### 2. 创建并执行 TaskRunner

- **约 79–99 行**：  
  - 用 `task_runner_class = ray.remote(TaskRunner)` 创建远程类；  
  - `runner = task_runner_class.remote()` 得到一个 Ray Actor；  
  - `ray.get(runner.run.remote(config))` 在 Ray 里执行 `TaskRunner.run(config)`，并等它跑完。

也就是说：**真正“干活”的是跑在 Ray 进程里的 `TaskRunner.run(config)`**。

---

## 三、TaskRunner.run(config)：建数据集、采样器、Trainer，再 fit

- **文件**：`verl/trainer/main_ppo.py`，类 `TaskRunner`，方法 `run`（约 269–355 行）。

### 1. 打印配置、解析

- **约 283–287 行**：打印 hostname、PID、完整 config（`OmegaConf.to_container(config, resolve=True)`），并 `OmegaConf.resolve(config)` 解析占位符。

### 2. 按 config 注册各类 Worker（不真正起进程，只登记“用哪类 Worker”）

- **约 289–296 行**：  
  - `add_actor_rollout_worker(config)`：根据 strategy（fsdp/fsdp2/megatron 等）和是否用新 engine，把 **ActorRollout（或 ActorRolloutRef）** 的 Ray Worker 类放进 `self.role_worker_mapping`。  
  - `add_critic_worker(config)`：登记 **Critic** Worker。  
  - `add_reward_model_resource_pool(config)`：若启用 reward model，登记 **RewardModel** 资源池。  
  - `add_ref_policy_worker(config, ...)`：若需要 reference policy（KL loss / KL in reward），登记 **RefPolicy** Worker。

### 3. 校验 config、下载模型、建 tokenizer/processor

- **约 298–315 行**：  
  - `validate_config(...)` 校验算法、actor、critic 等是否一致。  
  - `copy_to_local(config.actor_rollout_ref.model.path, ...)` 把模型路径拉到本地（若在 HDFS 等）。  
  - `hf_tokenizer(local_path, ...)`、`hf_processor(...)` 得到 **tokenizer** 和 **processor**（多模态用）。

### 4. 资源池与数据集、采样器

- **约 317–318 行**：`resource_pool_manager = self.init_resource_pool_mgr(config)`，按 `config.trainer.nnodes/n_gpus_per_node` 等建 GPU 资源池（如 `global_pool`、可选的 `reward_pool`）。
- **约 322–337 行**：  
  - `create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, ...)` → **train_dataset**；  
  - 同样方式建 **val_dataset**；  
  - `create_rl_sampler(config.data, train_dataset)` → **train_sampler**（RandomSampler / SequentialSampler 或 curriculum 等）。  
  - 数据集类由 `config.data` 决定（如 `get_dataset_class(data_config)`），在 `verl/utils/dataset/rl_dataset.py` 等位置。

### 5. 创建 RayPPOTrainer 并初始化 Worker、开始训练

- **约 340–354 行**：  
  - `trainer = RayPPOTrainer(config=config, tokenizer=..., processor=..., role_worker_mapping=..., resource_pool_manager=..., train_dataset=..., val_dataset=..., collate_fn=..., train_sampler=...)`  
    在 **driver 进程** 里创建 trainer 对象（不建 Ray Worker，只保存 config 和 dataset/dataloader 等）。  
  - `trainer.init_workers()`：按资源池和 `role_worker_mapping` **真正创建并 spawn 各 Role 的 Ray Worker**（Actor、Critic、RefPolicy、Reward 等），并做模型初始化、reward_loop_manager 等。  
  - `trainer.fit()`：**训练主循环**，直到结束。

下面分开说 **init_workers** 和 **fit** 里具体做了哪些计算、涉及哪些代码。

---

## 四、trainer.init_workers()：创建并初始化所有 Worker

- **文件**：`verl/trainer/ppo/ray_trainer.py`，方法 `init_workers`（约 691–约 850+ 行）。

### 1. 资源池与 Worker 类绑定

- **约 697–786 行**：  
  - `resource_pool_manager.create_resource_pool()` 创建资源池；  
  - 根据 config 为每个 role（ActorRolloutRef/ActorRollout、Critic、RefPolicy）构造 **RayClassWithInitArgs**（即“带初始化参数的 Ray 类”），放进 `resource_pool_to_cls`；  
  - 若用 colocate，`create_colocated_worker_cls` 把多个 role 打成一个 Worker 类，在同一批 GPU 上跑。

### 2. 真正 spawn Worker、初始化模型

- **约 776–805 行**：  
  - 对每个资源池调用 `ray_worker_group_cls(..., ray_cls_with_init=...).spawn(...)`，在 Ray 上 **启动进程/GPU**，加载模型、初始化；  
  - 得到 `actor_rollout_wg`、`critic_wg`（若用 critic）、`ref_policy_wg`（若用且未与 actor colocate）等。

### 3. Reward、Checkpoint、Rollout 等管理器

- **约 806–850+ 行**（具体行号以你本地为准）：  
  - 若用 critic：给 critic_wg 设 loss_fn、可能 `reset()` 等；  
  - 若用 ref policy 且独立进程：`ref_policy_wg.init_model()`；  
  - **RewardLoopManager**：`RewardLoopManager(...)` 创建 `self.reward_loop_manager`，用于后面 `_compute_reward_colocate` 里调 `compute_rm_score(batch)`（既可以是 reward model 打分，也可以是 rule-based 的 default_compute_score）；  
  - CheckpointManager、AsyncRolloutManager 等也会在这里或 fit 前初始化，用于权重同步、异步 rollout。

到这里，**所有“算力”都已就绪**，只等 `fit()` 里按 batch 驱动数据流。

---

## 五、trainer.fit()：一轮迭代在做什么、涉及哪些计算

- **文件**：`verl/trainer/ppo/ray_trainer.py`，方法 `fit`（约 1238 行起）。

### 1. 最外层：加载 checkpoint、可选预验证、步数/epoch 循环

- **约 1256–1270 行**：`_load_checkpoint()`、`checkpoint_manager.update_weights()`；若 `val_before_train` 为 True，跑一次 `_validate()` 并打日志。  
- **约 1278–1294 行**：`global_steps`、`total_training_steps`、`for epoch in range(...): for batch_dict in self.train_dataloader:`，即 **按 dataloader 的 batch 迭代**。

### 2. 每个 batch 的开头：从 dataloader 取数据、构造 DataProto、生成用的 gen_batch

- **约 1315–1321 行**：  
  - `batch = DataProto.from_single_dict(batch_dict)`，把 dataloader 的一个 batch 转成 **DataProto**；  
  - 设置 `batch.meta_info["temperature"]`；  
  - **为每条样本生成一个 uid**：`batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for ...])`（注意：这里每个样本一个 uid，后面 repeat 成 n 条 response 时，同一 prompt 的 n 条会共用一个 uid，在 GRPO/EGPO 里用于分组）。  
- **约 1315、1318–1321 行**：  
  - `gen_batch = self._get_gen_batch(batch)`（`ray_trainer.py` 约 491 行）：从 batch 里 **弹出与 reward 无关的 key**，得到只含生成所需字段的 `gen_batch`；  
  - `gen_batch_output = gen_batch.repeat(repeat_times=rollout.n, interleave=True)`，即每个 prompt 重复 n 份（例如 4 份），用于采样 n 条 response。

### 3. 生成 response（Rollout）

- **约 1325–1335 行**：  
  - `gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)`  
  - 这里通过 RPC 调用 **Actor Rollout Worker**，用当前策略对每个 prompt 采样出 **n 条 response**（tokens），结果写回 `gen_batch_output`（含 `input_ids`、`responses`、`attention_mask` 等）。  
- **约 1366–1367 行**：  
  - `batch = batch.repeat(..., interleave=True)` 与 rollout 的 n 对齐；  
  - `batch = batch.union(gen_batch_output)`，把 **prompt + 采样得到的 responses** 合并进同一个 batch。

### 4. response_mask、balance_batch、reward 计算

- **约 1369–1376 行**：若没有 `response_mask`，则 `compute_response_mask(batch)`；若开 `balance_batch`，则 `_balance_batch(batch, metrics=...)`。  
- **约 1387–1395 行**：  
  - 若用 reward model 且 batch 里还没有 `rm_scores`：`batch_reward = self._compute_reward_colocate(batch)`（内部调 `reward_loop_manager.compute_rm_score(batch)`），把结果 merge 进 batch；  
  - **extract_reward(batch)**（`verl/trainer/ppo/reward.py` 约 152 行）：从 batch 里取出 `reward_tensor = batch.batch["rm_scores"]` 和 `reward_extra_infos_dict`。  
- **约 1464–1477 行**（fit 里写回 batch 的位置）：  
  - `batch.batch["token_level_scores"] = reward_tensor`；  
  - 若 `use_kl_in_reward`，则 `apply_kl_penalty(...)` 得到 `token_level_rewards`；否则 `batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]`。  
也就是说：**reward 的来源是 RM 或 rule-based（通过 RewardLoopManager / default_compute_score），最终以 token 维度的分数形式放在 `token_level_scores`，再变成 `token_level_rewards`（可能减 KL 惩罚）。**

### 5. old_log_prob（以及 EGPO 时的熵）

- **约 1410–1440 行**：  
  - 若非 bypass 模式，需要 **recompute old_log_prob**：  
    - 若为 EGPO，先设 `batch.meta_info["calculate_entropy"] = True`；  
    - `old_log_prob, _ = self._compute_old_log_prob(batch)`（约 1146 行）：通过 RPC 调 **Actor** 的 `compute_log_prob(batch)`，得到 `old_log_probs` 和可选的 `entropys`；  
    - 若为 EGPO，**不**对 `old_log_prob.batch` 做 `pop("entropys")`，这样后面 `batch.union(old_log_prob)` 后 batch 里仍有 `entropys`；非 EGPO 则 pop 掉；  
  - `batch = batch.union(old_log_prob)`，合并进 batch。

### 6. reference log_prob、values（若用 critic）

- **约 1449–1459 行**：若用 reference policy，`ref_log_prob = self._compute_ref_log_prob(batch)`，合并进 batch；若用 critic，`values = self._compute_values(batch)`，合并进 batch。

### 7. 写 token_level_scores、算 token_level_rewards、compute_advantage

- **约 1461–1501 行**：  
  - 把前面得到的 reward 写进 `batch.batch["token_level_scores"]`，并按 config 决定是否做 KL 惩罚，得到 `token_level_rewards`；  
  - 若有 rollout_correction（decoupled 等），会在这里做 IS weight、rejection 等并写回 batch；  
  - **compute_advantage(batch, adv_estimator=config.algorithm.adv_estimator, ..., config=config.algorithm)**（`ray_trainer.py` 约 129 行）：  
    - 根据 `adv_estimator` 调用 **GAE / GRPO / EGPO** 等（在 `core_algos.py` 里实现），用 `token_level_rewards`、`response_mask`、`uid`（及 EGPO 时的 `entropys`）算出 **advantages** 和 **returns**，写回 `batch.batch["advantages"]`、`batch.batch["returns"]`；  
    - EGPO 分支用完后会 `data.batch.pop("entropys")`。

### 8. update_critic、update_actor

- **约 1508–1521 行**：  
  - 若用 critic：`critic_output = self._update_critic(batch)`（约 1207 行），通过 RPC 用当前 batch 的 returns/values 更新 Critic；  
  - 若步数超过 critic_warmup：`actor_output = self._update_actor(batch)`（约 1171 行），通过 RPC 把 batch（含 `advantages`、`old_log_probs`、`ref_log_prob` 等）发给 Actor，在 Worker 里算 **PPO policy loss**（clip 等），反向传播更新 Actor 参数。

### 9. 日志、checkpoint、验证、步数递增

- **约 1522 之后**：记录 metrics、logger.log、进度条更新、按配置做 checkpoint、是否做 validation；`self.global_steps += 1`，继续下一 batch 或下一 epoch，直到达到 `total_training_steps` 或总 epoch 结束。

---

## 六、数据与计算小结（单步 batch）

| 阶段           | 做什么                           | 主要代码位置 |
|----------------|----------------------------------|--------------|
| 取 batch       | dataloader → DataProto，加 uid   | `ray_trainer.py` fit 内 1315–1321 |
| 生成 response  | Actor Rollout 采样 n 条/ prompt  | `async_rollout_manager.generate_sequences` |
| 合并 rollout   | batch.repeat + union(gen_batch_output) | fit 内 1366–1367 |
| response_mask  | 算哪些 token 属于 response       | `compute_response_mask` |
| Reward         | RM 或 rule-based → rm_scores → token_level_scores/rewards | `_compute_reward_colocate`、`extract_reward`、fit 内 1464–1477 |
| old_log_prob   | Actor compute_log_prob，EGPO 时带熵 | `_compute_old_log_prob`、fit 内 1410–1440 |
| advantage      | GAE/GRPO/EGPO 用 rewards、uid、可选 entropys | `compute_advantage`、`core_algos.py` |
| update critic  | 用 returns/values 更新 value 网络 | `_update_critic` |
| update actor   | 用 advantages、old_log_probs 算 PPO loss 更新策略 | `_update_actor`、Worker 内 policy loss |

---

## 七、和 YAML 的对应关系

- **config-path / config-name**：决定从哪个 yaml 起加载，以及 `defaults:` 里引用的 algorithm、actor、rollout、data、reward 等子配置。
- **algorithm.adv_estimator**：在 `compute_advantage` 里分支，选 `gae` / `grpo` / `egpo` 等。
- **reward**：由 `reward` 和 `data`（如 `data_source`）等决定用 RM 还是 rule-based（如 `default_compute_score`），最终都落到 `rm_scores` → `token_level_scores` → `token_level_rewards`。

按上述顺序从 `main_ppo` → `run_ppo` → `TaskRunner.run` → `init_workers` → `fit` 的每一步对照代码看，就能把“从 main_ppo 到训练结束”的整条链路和涉及的计算串起来。
