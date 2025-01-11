set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export BASE_DIR=/mnt/bn/seed-rlhf-hl/zhangchi.usc1992

model_path=$BASE_DIR/models/Qwen2.5-3B-Instruct

n_samples=5

for split in "test" "train"
do
    output_path=${BASE_DIR}/data/gsm8k/Qwen2.5-3B-Instruct_${split}_output.parquet

    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=16 \
        data.path=$BASE_DIR/data/gsm8k/${split}.parquet \
        data.prompt_key=prompt \
        data.n_samples=$n_samples \
        data.output_path=${output_path} \
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

    output_ranking_path=${BASE_DIR}/data/gsm8k/Qwen2.5-3B-Instruct_${split}_output_after_ranking.parquet
    python3 examples/dpo_trainer/math/label_dataset.py --input_file=${output_path} --output_file=${output_ranking_path}
done
