nproc_per_node=16

export BASE_DIR=/mnt/bn/seed-rlhf-hl/zhangchi.usc1992

gsm8k_train_path=$BASE_DIR/data/gsm8k/train.parquet
gsm8k_test_path=$BASE_DIR/data/gsm8k/test.parquet

model_path=$BASE_DIR/models/Qwen2.5-3B-Instruct

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_dpo_trainer \
    data.train_files=$gsm8k_train_path \
    data.val_files=$gsm8k_test_path \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=8 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-2b-it \
    trainer.total_epochs=2 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null