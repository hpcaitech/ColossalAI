script_dir=$(cd "$(dirname "$0")" && pwd)
cd "${script_dir}"

# toy model, 2tp*2pp 1024, 128
python ./benchmark.py \
    --model="toy" \
    --dtype="fp16" \
    --batch_size=2 \
    --seq_len=1024 \
    --output_len=128 \
    --mb_size=1 \
    --pp_size=2 \
    --tp_size=2

# 7b, fp16, 2 gpu, 1024, 128
for BATCH_SIZE in 2 4 8 16; do
    python ./benchmark.py \
        --model="7b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=1024 \
        --output_len=128 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2 \
        --tp_size=2
done

# 7b, fp16, 2 gpu, 512, 512
for BATCH_SIZE in 2 4 8 16 32; do
    python ./benchmark.py \
        --model="7b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=512 \
        --output_len=512 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2 \
        --tp_size=2
done

# 7b, fp16, 2 gpu, 1024, 128
for BATCH_SIZE in 2 4 8; do
    python ./benchmark.py \
        --model="13b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=1024 \
        --output_len=128 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2 \
        --tp_size=2
done

# 13b, fp16, 2 gpu, 512, 512
for BATCH_SIZE in 2 4 8 16; do
    python ./benchmark.py \
        --model="13b" \
        --dtype="fp16" \
        --batch_size=${BATCH_SIZE} \
        --seq_len=512 \
        --output_len=512 \
        --mb_size=$((${BATCH_SIZE}/2)) \
        --pp_size=2 \
        --tp_size=2
done
