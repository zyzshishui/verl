nproc_per_node=16

export BASE_DIR=/mnt/bn/seed-rlhf-hl/zhangchi.usc1992

gsm8k_train_path=$BASE_DIR/data/gsm8k/Qwen2.5-3B-Instruct_train_output_after_ranking.parquet
gsm8k_test_path=$BASE_DIR/data/gsm8k/Qwen2.5-3B-Instruct_test_output_after_ranking.parquet

model_path=$BASE_DIR/models/Qwen2.5-3B-Instruct

save_path=$BASE_DIR/exp/Qwen2.5-3B-Instruct

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_dpo_trainer \
    data.train_files=$gsm8k_train_path \
    data.val_files=$gsm8k_test_path \
    data.train_batch_size=256 \
    data.micro_batch_size=8 \
    model.partial_pretrain=$model_path \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-dpo \
    trainer.experiment_name=gsm8k-dpo-Qwen2.5-3B-Instruct \
    trainer.total_epochs=10 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null