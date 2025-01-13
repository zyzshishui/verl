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
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, logprobs_from_logits
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import RMDataset
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking

from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


class FSDPDPOTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        # build tokenizer first
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        if self.config.data.chat_template is not None:
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        self._build_dataloader()
        # build model
        self.fsdp_model = self._build_model(self.config.model)
        self.optimizer, self.lr_scheduler = self._build_optimizer(self.config.optim, self.fsdp_model)

        self.fsdp_ref_model = self._build_model(self.config.ref_model)

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0
        assert self.config.data.micro_batch_size % dp_size == 0

        self.config.data.train_batch_size //= dp_size
        self.config.data.micro_batch_size //= dp_size

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = RMDataset(parquet_files=config.data.train_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=config.data.prompt_key,
                                       responses_key=config.data.response_key,
                                       max_length=config.data.max_length,
                                       add_eos=True)

        self.val_dataset = RMDataset(parquet_files=config.data.val_files,
                                     tokenizer=self.tokenizer,
                                     prompt_key=config.data.prompt_key,
                                     responses_key=config.data.response_key,
                                     max_length=config.data.max_length,
                                     add_eos=True)

        # build dataloader
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           drop_last=True)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=True,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.micro_batch_size,
                                         sampler=self.val_sampler,
                                         drop_last=True)

    def _build_model(self, config):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_local_path_from_hdfs(src=config.partial_pretrain, verbose=True)

        if config.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = config.trust_remote_code
        # load config first
        model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context():
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                          config=model_config,
                                                                          torch_dtype=torch.float32,
                                                                          attn_implementation='flash_attention_2',
                                                                          trust_remote_code=trust_remote_code)

        if config.get('enable_gradient_checkpointing', False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(model, config=config.fsdp_config.wrap_policy)
        if torch.distributed.get_rank() == 0:
            print(auto_wrap_policy)
        
        cpu_offload = CPUOffload(offload_params=config.fsdp_config.cpu_offload)

        fsdp_model = FSDP(module=model,
                          auto_wrap_policy=auto_wrap_policy,
                          param_init_fn=init_fn,
                          sharding_strategy=ShardingStrategy.FULL_SHARD,
                          mixed_precision=mixed_precision,
                          device_mesh=self.device_mesh,
                          sync_module_states=True,
                          device_id=torch.cuda.current_device(),
                          cpu_offload=cpu_offload,
                          use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        return fsdp_model

    def _build_optimizer(self, config, fsdp_model):
        optimizer = optim.AdamW(fsdp_model.parameters(),
                                lr=config.lr,
                                betas=config.betas,
                                weight_decay=config.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {total_steps}'
            )

        num_warmup_steps = int(total_steps * config.warmup_steps_ratio)

        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=total_steps)
        
        return optimizer, lr_scheduler


    def _compute_loss(self, batch):
        response_mask = batch.pop('response_mask')[:, :, :-1].cuda()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size, num_responses, seqlen = input_ids.size()

        input_ids = input_ids.view(-1, seqlen)
        attention_mask = attention_mask.view(-1, seqlen)

        labels = input_ids[:, 1:].cuda()

        # prepare for packed inputs. input_ids, position_ids
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.fsdp_model(input_ids=input_ids,  # (bsz * num_responses, seqlen)
                                     attention_mask=attention_mask,
                                     use_cache=False)  # prevent model thinks it it generating

        logits = output.logits # (batch_size * num_responses, seqlen, vocab_size)

        shift_logits = logits[..., :-1, :].contiguous()

        log_probs = logprobs_from_logits(logits, labels)  # (bsz * num_responses, seqlen - 1)
        log_probs = log_probs.view(batch_size, num_responses, -1)
        log_probs = log_probs * response_mask # (bsz, num_responses, seqlen - 1)
        log_probs = log_probs.sum(-1)

        if torch.distributed.get_rank() == 0:
            from IPython import embed
            embed()
        torch.distributed.barrier()

        chosen_logprobs = log_probs.view(batch_size, num_responses, 1)
        rejected_logprobs = log_probs.view(batch_size, 1, num_responses)

        pairwise_logprobs = chosen_logprobs - rejected_logprobs  # (bsz, N, N)

        scores = batch.pop('scores') # (bsz, N)
        mask = scores.view(batch_size, N, 1) - scores.view(batch_size, 1, N)
        positive_mask = (mask > 0).to(torch.float32)
        negative_mask = (mask < 0).to(torch.float32)
        loss_mask = positive_mask - negative_mask

        # construct mask using scores. Only the lower left has value -1, 0, 1. 1

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * loss_mask

        valid_token_this_rank = torch.sum(loss_mask)

        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
            dp_size = torch.distributed.get_world_size()
        else:
            dp_size = 1

        loss = torch.sum(loss) / valid_token_this_rank * dp_size  # possible bugs here for dp
        return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size)
        n_micro_batches = len(micro_batches)
        for micro_batch in micro_batches:
            loss = self._compute_loss(batch=micro_batch) / n_micro_batches
            loss.backward()

        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage('Before optimizer step', logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        # TODO: all reduce to get accurate loss
        return {'train/loss': loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss(batch)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        global_step = 0

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                # average metric across dp_ranks
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                global_step += 1

            # validation
            val_losses = []
            for data in self.val_dataloader:
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                # average metric across dp_ranks
                metric = {'val/loss': val_loss.detach().item()}
                tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='dpo_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('dp',))
    trainer = FSDPDPOTrainer(config=config, device_mesh=device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
