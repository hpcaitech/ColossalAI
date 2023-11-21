ROOT=$(realpath $(dirname $0))
PY_SCRIPT=${ROOT}/benchmark_gptq_llama.py
GPU=$(nvidia-smi -L | head -1 | cut -d' ' -f4 | cut -d'-' -f1)

mkdir -p logs

# benchmark llama2-7b one single GPU
for bsz in 16 32 64; do
    python3 ${PY_SCRIPT} --gptq_model ./llama2-7b-4bit --bench_type "auto-gptq" --tp_size 1 --pp_size 1 -b $bsz -s 512  --output_len 256 | tee logs/${GPU}_${bsz}_512.txt
done


for bsz in 16 32 64; do
    python3 ${PY_SCRIPT} --gptq_model ./llama2-7b-4bit --bench_type "auto-gptq"  --tp_size 1 --pp_size 1 -b $bsz -s 1024 --output_len 256 | tee logs/${GPU}_${bsz}_1024.txt
done
