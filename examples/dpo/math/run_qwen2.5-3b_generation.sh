export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_DIR=/mnt/bn/seed-rlhf-hl/zhangchi.usc1992

gsm8k_train_path=$BASE_DIR/data/gsm8k/train.parquet
gsm8k_test_path=$BASE_DIR/data/gsm8k/test.parquet

model_path=$BASE_DIR/models/Qwen2.5-3B-Instruct

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=16 \
    data.path=${gsm8k_train_path} \
    data.prompt_key=prompt \
    data.n_samples=5 \
    data.output_path=${BASE_DIR}/data/gsm8k/Qwen2.5-3B-Instruct_output.parquet \
    data.batch_size=2048 \
    model.path=${model_path} \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=-1 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8
