CUDA 12.4 uv Environment
========================

This repository carries a legacy-compatible CUDA 12.4 environment for the
online-sampling experiment. The current general VeRL installation guide targets
newer CUDA releases; this lock is intentionally narrower and must be validated
on the target GPU node.

The pinned core stack is:

.. code-block:: text

   Python 3.10
   PyTorch 2.6.0+cu124
   vLLM 0.8.5.post1
   FlashAttention 2.7.4.post1
   FlashInfer 0.2.2.post1+cu124torch2.6
   Transformers 4.57.6
   TransferQueue 0.1.7

Create the environment with either a local BFCL checkout or a pre-built BFCL
wheel:

.. code-block:: bash

   BFCL_SOURCE=/data/zhangdw12/work/gorilla/berkeley-function-call-leaderboard \
     bash scripts/bootstrap_verl_cu124.sh

.. code-block:: bash

   BFCL_WHEEL=/path/to/bfcl_eval.whl \
     bash scripts/bootstrap_verl_cu124.sh

The script installs ``requirements-cu124.lock`` first, then installs the current
VeRL checkout with ``uv pip install --no-deps -e .`` so dependency resolution
cannot replace the pinned CUDA stack.

Server layout
-------------

The default virtual environment path is:

.. code-block:: text

   /data/zhangdw12/work/uv-venv/verl-cu124

Override it when necessary:

.. code-block:: bash

   VENV_PATH=/another/path/verl-cu124 \
   BFCL_SOURCE=/data/zhangdw12/work/gorilla/berkeley-function-call-leaderboard \
     bash scripts/bootstrap_verl_cu124.sh

The same environment can run VeRL training, the vLLM evaluation server, and
the BFCL evaluator. ``BFCL_SOURCE`` should point to the required Gorilla/BFCL
checkout; ``BFCL_WHEEL`` is intended for a wheel built from an equivalent
checkout.

Validation performed by the script
----------------------------------

The bootstrap command:

1. rejects non-Linux or non-x86_64 hosts;
2. installs the fully resolved CUDA 12.4 lock;
3. installs the current VeRL checkout without dependency resolution;
4. installs the selected BFCL runtime;
5. runs ``uv pip check``;
6. imports PyTorch, vLLM, FlashAttention, FlashInfer, TransferQueue, VeRL, and
   the BFCL multi-turn loader/checker modules;
7. verifies that CUDA is available and ``torch.version.cuda == "12.4"``.

The final GPU/import check must be run on the target server; a CPU-only
development machine cannot validate the CUDA wheels or H20 execution path.
