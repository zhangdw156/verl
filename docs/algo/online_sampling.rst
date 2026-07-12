Persistent Online Sampling
==========================

``main_ppo_sync`` supports an optional persistent prompt sampler for GRPO.
The mode performs a resumable full-pool Step 0 probe, then samples prompts
using reward-variance and staleness weights. Prompt groups with near-zero
unnormalized reward variance are recorded but removed before the optimizer
update.

Dataset contract
----------------

The training dataset must expose:

.. code-block:: python

   dataset.sample_ids          # stable sequence aligned with __getitem__
   dataset.dataset_fingerprint # stable source/catalog fingerprint
   dataset.sample_metadata     # optional JSON-serializable metadata

Each item must include ``raw_prompt``, ``reward_model``, ``index``, and
``sample_id``. The custom reward function must return:

.. code-block:: python

   {
       "score": float_value,
       "correct": 0_or_1,
       "format": 0_or_1,
   }

Configuration
-------------

.. code-block:: yaml

   data:
     train_batch_size: 128
     gen_batch_size: 160
     train_custom_cls:
       path: /path/to/dataset.py
       name: CustomDataset

   algorithm:
     adv_estimator: grpo
     online_sampling:
       enabled: true
       state_path: /persistent/path/online_sampling.sqlite
       rollout_n: 8
       zero_variance_epsilon: 1.0e-6
       ema_alpha: 0.8
       min_weight: 0.001
       staleness_weight: 0.1
       staleness_horizon: 100
       max_refill_rounds: 5
       min_effective_batch_ratio: 0.25
       probe_batch_size: 128
       seed: 42

   actor_rollout_ref:
     rollout:
       n: 8

   trainer:
     total_training_steps: 184

``trainer.total_training_steps`` is required because online weighted sampling
does not have epoch semantics. SQLite uses WAL mode and stores raw rollout
outcomes, aggregate observations, latest EMA state, and checkpoint-aligned
snapshots.

Priority and filtering semantics
--------------------------------

For sample ``i``, the sampling weight is:

.. math::

   w_i = w_{\min}
       + \operatorname{clip}\left(\frac{\operatorname{EMA}(\sigma_i)}
                                      {\sigma_{\mathrm{scale}}}, 0, 1\right)
       + \lambda_s \min\left(\frac{\mathrm{age}_i}{H_s}, 1\right)

``ema_alpha`` gives the newest reward standard deviation the dominant weight.
``sigma_scale`` is the 95th percentile of positive Step 0 standard deviations.
Every sample retains a positive ``min_weight`` and therefore remains eligible
for later re-evaluation.

After generation, a complete prompt group is recorded before filtering. Groups
with ``sigma <= zero_variance_epsilon`` do not enter the optimizer update.
The trainer samples replacements for at most ``max_refill_rounds`` generation
rounds. If the final effective prompt count is at least
``min_effective_batch_ratio * train_batch_size``, existing padding utilities
allow the smaller batch to train; otherwise the optimizer step is skipped.

State and recovery
------------------

Step 0 probes every eligible sample exactly once unless interrupted. Completed
groups are committed to SQLite immediately, so a restart processes only the
remaining sample IDs. The mode does not perform periodic full-pool reprobes.

The run fingerprint includes the ordered sample IDs, dataset fingerprint,
model manifest, reward and BFCL handler hashes, rollout parameters, and every
online-sampling value that changes persistent state semantics. Reusing a
``run_id`` with a different fingerprint fails closed.

Regular checkpoints store a named SQLite snapshot before publishing
``latest_checkpointed_iteration.txt``. Resume restores the model and the
matching active sampler snapshot; post-checkpoint observations remain in the
append-only history for audit but do not affect the restored active state.

Useful metrics
--------------

The online loop emits:

.. code-block:: text

   online_sampling/candidate_prompts
   online_sampling/effective_prompts
   online_sampling/zero_variance_prompts
   online_sampling/generation_rounds
   online_sampling/refill_rounds
   online_sampling/effective_batch_ratio
   online_sampling/skipped_optimizer_step
   online_sampling/db_commit_seconds
   online_sampling/priority_sigma_mean
   online_sampling/staleness_mean

Best validation checkpoint
--------------------------

The synchronous trainer can retain a metric-selected checkpoint independently
of recent-checkpoint rotation:

.. code-block:: yaml

   trainer:
     best_checkpoint:
       enabled: true
       metric: val-core/bfcl_v4_multi_turn/overall/accuracy
       mode: max
       min_delta: 0.0
       directory: best_checkpoint

Training resume continues to use the latest regular checkpoint. The best
checkpoint is materialized from a completed regular checkpoint using hard links
when possible, so it is not registered in recent-checkpoint rotation. Best
checkpoint retention requires synchronous actor and critic checkpoint saving;
set ``checkpoint.async_save=false``. The current implementation supports the
standard ``actor/`` checkpoint layout and fails closed for TorchTitan.

The selected step is recorded at:

.. code-block:: text

   <trainer.default_local_dir>/<directory>/best_checkpointed_iteration.txt

The corresponding actor checkpoint, optional critic checkpoint, dataloader
state, ``best_metric.json``, and named SQLite snapshot are retained together.
Training resume still follows the regular latest-checkpoint pointer.
