# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ray
import os

import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig, FullStateDictConfig, FullOptimStateDictConfig

from verl.utils.fs import copy_local_path_from_hdfs, is_non_local

from transformers import PreTrainedTokenizer

from .checkpoint_manager import BaseCheckpointManager

try:
    import hdfs_io
except ImportError:
    import verl.utils.hdfs_io as hdfs_io


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save 
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(self,
                 model: FSDP,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 tokenizer: PreTrainedTokenizer,
                 sharded_state_dict: bool = True,
                 *args,
                 **kwargs):
        super().__init__(model, optimizer, lr_scheduler, tokenizer)
        self.sharded_state_dict = sharded_state_dict

    def load_checkpoint(self, path=None, del_local_after_load=True, *args, **kwargs):
        if path is None:
            return

        # Load extra state
        remote_extra_state_path = os.path.join(path, f'extra_state_rank_{self.rank}.pt')
        print(f'[rank-{self.rank}]: Loading extra state from {remote_extra_state_path}')
        local_extra_state_path = copy_local_path_from_hdfs(remote_extra_state_path)
        extra_state_dict = torch.load(local_extra_state_path)
        if del_local_after_load:
            try:
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )
        lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

        # Recover random state
        if 'rng' in extra_state_dict:
            self.load_rng_state(extra_state_dict['rng'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        if self.sharded_state_dict:
            # Sharded loading
            remote_model_path = os.path.join(path, f'model_rank_{self.rank}.pt')
            remote_optim_path = os.path.join(path, f'optim_rank_{self.rank}.pt')
            print(f'[rank-{self.rank}]: Loading model and optim from {remote_model_path} and {remote_optim_path}')
            local_model_path = copy_local_path_from_hdfs(remote_model_path)
            local_optim_path = copy_local_path_from_hdfs(remote_optim_path)

            model_state_dict = torch.load(local_model_path)
            optimizer_state_dict = torch.load(local_optim_path)

            if del_local_after_load:
                try:
                    os.remove(local_model_path) if is_non_local(local_model_path) else None
                    os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                except Exception as e:
                    print(
                        f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                    )

            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                self.model.load_state_dict(model_state_dict)
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(optimizer_state_dict)
        else:
            # Load full checkpoint on rank 0 and broadcast
            model_state_dict = None
            optimizer_state_dict = None

            if self.rank == 0:
                remote_model_path = os.path.join(path, 'model.pt')
                remote_optim_path = os.path.join(path, 'optim.pt')
                print(f'[rank-{self.rank}]: Loading model and optim from {remote_model_path} and {remote_optim_path}')

                local_model_path = copy_local_path_from_hdfs(remote_model_path)
                local_optim_path = copy_local_path_from_hdfs(remote_optim_path)

                model_state_dict = torch.load(local_model_path, map_location="cpu")
                optimizer_state_dict = torch.load(local_optim_path, map_location="cpu")

                if del_local_after_load:
                    try:
                        os.remove(local_model_path) if is_non_local(local_model_path) else None
                        os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                    except Exception as e:
                        print(
                            f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                        )

            # Create lists for broadcasting
            model_state_list = [model_state_dict]
            optim_state_list = [optimizer_state_dict]

            # Broadcast model state dict
            torch.distributed.broadcast_object_list(model_state_list, src=0)
            model_state_dict = model_state_list[0]  # Get the broadcasted value

            # Broadcast optimizer state dict
            torch.distributed.broadcast_object_list(optim_state_list, src=0)
            optimizer_state_dict = optim_state_list[0]  # Get the broadcasted value

            # Load model and optimizer
            with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(rank0_only=True),
                    optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True),
            ):
                # Load model
                self.model.load_state_dict(model_state_dict)

                # Load optimizer
                if self.optimizer is not None:
                    # Convert full optimizer state dict to sharded format
                    sharded_optim_state = FSDP.optim_state_dict_to_load(self.model, self.optimizer,
                                                                        optimizer_state_dict)
                    self.optimizer.load_state_dict(sharded_optim_state)

    def save_checkpoint(self,
                        local_path: str,
                        global_step: int,
                        hdfs_path: str = None,
                        remove_previous_ckpt=True,
                        *args,
                        **kwargs):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # saving extra state
        if self.lr_scheduler is not None:
            lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        else:
            lr_scheduler_state_dict = None

        extra_state_dict = {
            'lr_scheduler': lr_scheduler_state_dict,
            'rng': self.get_rng_state(),
        }
        extra_path = os.path.join(local_path, f'extra_state_rank_{self.rank}.pt')
        print(f'[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}')
        torch.save(extra_state_dict, extra_path)

        # save sharded state dict
        if self.sharded_state_dict:
            # every rank will save its own model and optim shard
            state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                    model_state_dict = self.model.state_dict()
                    if self.optimizer is not None:
                        optimizer_state_dict = self.optimizer.state_dict()
                    else:
                        optimizer_state_dict = None

                    model_path = os.path.join(local_path, f'model_rank_{self.rank}.pt')
                    optim_path = os.path.join(local_path, f'optim_rank_{self.rank}.pt')

                    print(f'[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}')
                    print(f'[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}')
                    torch.save(model_state_dict, model_path)
                    torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None

                if self.rank == 0:
                    hf_local_path = os.path.join(local_path, 'huggingface')
                    os.makedirs(hf_local_path, exist_ok=True)
                    self.model._fsdp_wrapped_module.config.save_pretrained(hf_local_path)
                    self.tokenizer.save_pretrained(hf_local_path)
        else:
            state_dict_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_cfg):
                state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                    optimizer_state_dict = FSDP.optim_state_dict(self.model, self.optimizer, optimizer_state_dict)
                else:
                    optimizer_state_dict = None

                model_path = os.path.join(local_path, f'model.pt')
                optim_path = os.path.join(local_path, f'optim.pt')

            if self.rank == 0:
                # save for resume
                torch.save(state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)  # TODO: address optimizer is None
                torch.save(extra_state_dict, extra_path)
                # save for huggingface
                hf_local_path = os.path.join(local_path, 'huggingface')
                print(f'Saving model to {hf_local_path}')
                print(f'Saving model to {optim_path}')
                os.makedirs(hf_local_path, exist_ok=True)
                self.model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(hf_local_path)

                # TODO (sgm): backward compatible, delete it when uploader is ready
                if hdfs_path is not None:
                    print(f'Uploading actor checkpoint to {hdfs_path}')
                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=local_path, dst=hdfs_path, dirs_exist_ok=True)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        self.previous_save_local_path = local_path
