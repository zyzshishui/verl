# Tested in 4 GPUs

set -x

nproc_per_node=$1

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size=32 \
    model.partial_pretrain=$HOME/models/SmolLM2-135M-Instruct \
    trainer.default_hdfs_dir=$HOME/results/SmolLM2-135M-Instruct \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-SmolLM2-135M \
    trainer.total_epochs=3 \
    trainer.logger=['console','tracking']