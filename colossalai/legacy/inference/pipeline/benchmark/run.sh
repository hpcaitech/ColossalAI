script_dir=$(cd "$(dirname "$0")" && pwd)
cd "${script_dir}"

# 7b, fp16, 2 gpu, 1024, 128
for BATCH_SIZE in 2 4 8 16; do
    CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
        --model="7b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=1024 \
        --new_length=128 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2
done

# 7b, fp16, 2 gpu, 512, 512
for BATCH_SIZE in 2 4 8 16 32; do
    CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
        --model="7b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=512 \
        --new_length=512 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2
done

# 7b, fp16, 2 gpu, 1024, 128
for BATCH_SIZE in 2 4 8; do
    CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
        --model="13b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=1024 \
        --new_length=128 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2
done

# 13b, fp16, 2 gpu, 512, 512
for BATCH_SIZE in 2 4 8 16; do
    CUDA_VISIBLE_DEVICES=0,1 colossalai run --nproc_per_node 2 --master_port 29800 ./benchmark.py \
        --model="13b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=512 \
        --new_length=512 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2
done
