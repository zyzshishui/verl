set -x

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

python3 -m verl.trainer.main_ppo --config-path=/root/verl/examples/sglang_multiturn/config --config-name='gsm8k_multiturn'