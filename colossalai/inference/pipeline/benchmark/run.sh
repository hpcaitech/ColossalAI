script_dir=$(cd "$(dirname "$0")" && pwd)
cd "${script_dir}"

CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
    --model="7b" \
    --fp16 \
    --batch_size=2 \
    --seq_len=1024 \
    --new_length=128 \
    --mb_size=1 \
    --pp_size=2

CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
    --model="7b" \
    --fp16 \
    --batch_size=4 \
    --seq_len=1024 \
    --new_length=128 \
    --mb_size=2 \
    --pp_size=2

CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
    --model="7b" \
    --fp16 \
    --batch_size=8 \
    --seq_len=1024 \
    --new_length=128 \
    --mb_size=4 \
    --pp_size=2

CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
    --model="7b" \
    --fp16 \
    --batch_size=16 \
    --seq_len=1024 \
    --new_length=128 \
    --mb_size=8 \
    --pp_size=2
