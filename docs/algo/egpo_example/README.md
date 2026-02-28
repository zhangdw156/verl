# 从你的 GRPO 配置切换到 EGPO

## 1. 文件放置

把这些文件拷到你当前工程里，**保持 YAML 里的路径与你的实际路径一致**（按需改 `train_files`、`val_files`、`model.path`、`custom_reward_function.path` 等）。

建议：

| 本目录文件 | 拷贝到（示例） |
|-----------|----------------|
| `configs/verl_common_config_egpo.yaml` | `/dfs/data/work/hardtry/exps/verl6/configs/verl_common_config_egpo.yaml` |
| `configs/egpo_config.yaml` | `/dfs/data/work/hardtry/exps/verl6/configs/egpo_config.yaml` |
| `reward_fn_egpo.py` | `/dfs/data/work/hardtry/src/hardtry/rl/reward_fn_egpo.py` |

若你拷贝到的路径不同，请修改：

- **verl_common_config_egpo.yaml** 里的 `custom_reward_function.path`，指向你放 `reward_fn_egpo.py` 的路径。
- **egpo_config.yaml** 里 `hydra.searchpath` 的 `file:///...`，指向你放「verl_common_config_egpo.yaml」的目录（和 grpo 时一致即可，例如 `file:///dfs/data/work/hardtry/exps/verl6/configs/`）。

## 2. 与 GRPO 的差异

- **algorithm**：`adv_estimator: egpo`，并增加 `egpo_lambda: 0.4`、`egpo_alpha: 2.0`。
- **reward**：EGPO 使用**严格二元**：只有「格式正确（含 `<tool_call>`）且解析结果与 ground_truth 一致」为 1.0，其余为 0.0；你的 GRPO 是 0 / 0.1 / 1.0 / 1.1。
- **entropy**：`entropy_coeff` 仍为 0；EGPO 的熵只用于 advantage 计算，由 trainer 在算 old_log_prob 时自动打开。

## 3. 启动命令

在工程根目录（或你平时跑 GRPO 的目录）执行，指定 **config-path** 和 **config-name=egpo_config**：

```bash
python3 -m verl.trainer.main_ppo \
  --config-path=/dfs/data/work/hardtry/exps/verl6/configs \
  --config-name=egpo_config
```

若你用脚本跑，把原来的 `--config-name=grpo_config` 换成 `--config-name=egpo_config`，并保证 `--config-path` 指向包含 `egpo_config.yaml` 和 `verl_common_config_egpo.yaml` 的目录。

## 4. 可选：共用一套 common config

若不想维护两份 common，可以只保留一个 `verl_common_config.yaml`，在里面对 algorithm 用默认 grpo，然后单独写一个只覆盖 algorithm 和 custom_reward_function 的 yaml（例如 `egpo_override.yaml`），在 `egpo_config.yaml` 的 defaults 里写：

```yaml
defaults:
  - ppo_trainer
  - verl_common_config
  - egpo_override   # 只覆盖 algorithm 和 custom_reward_function
  - _self_
```

这样 GRPO 用 `grpo_config`，EGPO 用 `egpo_config` 即可。
