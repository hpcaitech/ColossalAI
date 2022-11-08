export DATA=/data/scratch/gpt_data/small-gpt-dataset.json

export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}

env OMP_NUM_THREADS=16 torchrun --standalone --nproc_per_node=2 train_gpt.py --config=gpt2_configs/gpt2_zero3.py --from_torch 2>&1 | tee logs/log
