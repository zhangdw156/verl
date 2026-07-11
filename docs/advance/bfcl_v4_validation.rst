BFCLv4 Multi-Turn Validation
============================

Last updated: 07/11/2026.

VeRL can run the complete BFCLv4 multi-turn suite during training with the
current rollout weights. The validation path does not export checkpoints or
start a second inference server. Users provide only:

1. the raw BFCLv4 data directory; and
2. one independent model-specific handler file.

Requirements
------------

Install the ``bfcl_eval`` package on every Ray worker. For a local Gorilla
checkout, an editable no-dependency install keeps BFCL's large optional API
dependencies out of the base VeRL environment:

.. code-block:: bash

   uv pip install --no-deps -e /path/to/gorilla/berkeley-function-call-leaderboard
   uv pip install openai overrides tenacity mpmath python-dotenv \
     tree_sitter==0.21.3 tree-sitter-java==0.21.0 tree-sitter-javascript==0.21.4

The raw data root must contain:

.. code-block:: text

   BFCL_v4_multi_turn_base.json
   BFCL_v4_multi_turn_miss_func.json
   BFCL_v4_multi_turn_miss_param.json
   BFCL_v4_multi_turn_long_context.json
   possible_answer/BFCL_v4_multi_turn_*.json
   multi_turn_func_doc/*.json

Handler file
------------

The handler file only needs to be visible to the driver. Before Ray starts,
VeRL embeds the source and its SHA256 checksum into the distributed runtime
configuration, so workers do not need a matching absolute filesystem path.
By default, VeRL loads an object named ``Handler`` from the file. The easiest
option is a zero-argument subclass of an existing BFCL prompting handler:

.. literalinclude:: ../../examples/bfcl_v4_validation/qwen3_handler.py
   :language: python

The exported object must implement the BFCL prompting-handler methods used by
the official multi-turn loop:

.. code-block:: text

   _pre_query_processing_prompting
   add_first_turn_message_prompting
   _add_next_turn_user_message_prompting
   _format_prompt
   _parse_query_response_prompting
   _add_assistant_message_prompting
   _add_execution_results_prompting
   decode_execute

Configuration
-------------

.. code-block:: yaml

   data:
     val_files: /path/to/bfcl_eval/data
     val_batch_size: 16
     validation_shuffle: false
     bfcl_v4:
       enabled: true
       data_root: /path/to/bfcl_eval/data
       handler:
         path: /shared/handlers/my_handler.py
         name: Handler
         kwargs: {}

   actor_rollout_ref:
     rollout:
       val_kwargs:
         n: 1
         do_sample: false
         temperature: 0

   trainer:
     val_before_train: true
     test_freq: 10

The dataset loader validates all four official categories and expects 200
examples per category. Configuration, dependency, handler-contract, or data
integrity errors abort validation. Invalid model calls, decode failures, and
step-limit termination receive zero accuracy with diagnostic metrics.

Metrics
-------

VeRL reports category accuracy and the official unweighted multi-turn overall:

.. code-block:: text

   val-core/bfcl_v4/multi_turn_base/acc/mean@1
   val-core/bfcl_v4/multi_turn_miss_func/acc/mean@1
   val-core/bfcl_v4/multi_turn_miss_param/acc/mean@1
   val-core/bfcl_v4/multi_turn_long_context/acc/mean@1
   val-core/bfcl_v4_multi_turn/overall/accuracy

The BFCL AgentLoop is validation-only. Model generations always go through
VeRL's ``server_manager.generate()``, so evaluation uses the weights currently
loaded in the rollout workers. Before calling BFCL's executor, VeRL validates
every decoded function-call AST against the exposed function names and permits
only literal arguments.
