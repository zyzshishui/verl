Using Checkpoints to Support Fault Tolerance Training
=====================================================

There could be training errors or machine failure during the whole RLHF training process, 
so it is recommended to enable checkpoints to minimize your loss.

The API Interface has already been listed in :ref:`config-explain-page`,
and we will not repeat them. But there are still some technique details
we hope to clarify.

.. note:: 

    Notice that the ``checkpoint.contents`` field has no effect to FSDP checkpoint except ``hf_model``, 
    the other 3 fields are binded together to save and load. We recommend to include ``model``, ``optimizer`` and ``extra`` all.

Checkpoint Saving Directory Structure
-------------------------------------

Commonly, we use the ``default_local_dir`` declared in ``ppo_trainer.yaml`` or ``ppo_megatron_trainer.yml``
to work as preffix when saving checkpoints, which is ``checkpoints/${trainer.project_name}/${trainer.experiment_name}``.

So the inner checkpoint structure of **FSDP** is like:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   ├── actor_huggingface
    │   ├── critic
    │   │   ├── model_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   ├── optim_world_size_{self.world_size}_rank_{self.rank}.pt
    │   │   └── extra_state_world_size_{self.world_size}_rank_{self.rank}.pt
    │   └── critic_huggingface
    └── latest_checkpointed_iteration.txt

All model shards, optimizers and extra states are stored togather, in a sharded and distributed way.

While Megatron current checkpoint structure is:

.. code::

    checkpoints/${trainer.project_name}/${trainer.experiment_name}
    ├── global_steps_${i}
    │   ├── actor
    │   │   ├── huggingface     # default save tokenizer, save huggingface model if include ``hf_mode`` in checkpoint.contents
    │   │   ├── model           # save sharded model, naming the same as Megatron
    │   │   │   ├── mp_rank_xx_yyy          # xx is tp_rank in 2 digits, yyy is pp_rank in 3 digits
    │   │   │   │   └── model_states.pt
    │   │   │   └── mp_rank_xx_xxx
    │   │   ├── optim
    │   │   │   ├── distrib_optim_pp{x}_tp{y}.pt
    │   │   │   └── distrib_optim_pp{x}_tp{y}.pt
    │   │   └── rng_states
    │   └── critic
    │   │   ├── huggingface
    │   │   ├── model
    │   │   ├── optim
    │   │   └── rng_states
    └── latest_checkpointed_iteration.txt

Convert FSDP and Megatron Checkpoints to HuggingFace Format Model
-----------------------------------------------------------------

We provide a tool to convert the FSDP and Megatron checkpoints to HuggingFace format model.
The tool is located in ``scripts/model_merger.py``.

The arguments are as follows:

.. code:: bash

    usage: model_merger.py [-h] [--backend {fsdp,megatron}]
                           [--hf_model_path $original_model_path, like {Qwen/Qwen2-7B}]
                           [--local_dir $local_directory saved fsdp or megatron models]
                           [--target_dir $target_dir to save converted models, default is tmp]
                           [--hf_upload_path $huggingface_repo to upload]

So example use of Megatron model merger is:

.. code:: bash

    python3 scripts/model_merger.py --backend megatron \
        --hf_model_path Qwen/Qwen2-7B \
        --local_dir checkpoints/verl_megatron_gsm8k_examples/deepseek_megatron_checkpoint_saveload/global_step_1/actor/model

Megatron Merger details
-----------------------

Current implement of decoder layers use ``nn.ModuleList`` to store the layers, but not modify