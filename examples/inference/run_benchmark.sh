ROOT=$(realpath $(dirname $0))
PY_SCRIPT=${ROOT}/benchmark_llama.py
GPU=$(nvidia-smi -L | head -1 | cut -d' ' -f4 | cut -d'-' -f1)

mkdir -p logs

# benchmark llama2-7b one single GPU
for bsz in 16 32 64; do
    python3 ${PY_SCRIPT} -m llama2-7b --tp_size 1 --pp_size 1 -b $bsz -s 256 --output_len 128 | tee logs/${GPU}_${bsz}_256.txt
done


for bsz in 4 8 16 32 64; do
    python3 ${PY_SCRIPT} -m llama2-7b --tp_size 1 --pp_size 1 -b $bsz -s 1024 --output_len 128 | tee logs/${GPU}_${bsz}_1024.txt
done
