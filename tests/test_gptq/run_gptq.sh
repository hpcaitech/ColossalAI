# OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=5 python quant_llama.py /data/scratch/llama-7b-hf c4 \
# 	--wbits 4 --true-sequential --groupsize 128 --save ./llama7b-4bit-128g-cai-nao.pt\
# 	--benchmark --model_type cai  --input_len 1024 --max_new_tokens 128 --batch_size 1

# OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=5 python quant_llama.py /data/scratch/llama-7b-hf c4 \
# 	--wbits 4 --true-sequential --groupsize 128 --save ./llama7b-4bit-128g-gptq-nao.pt\
# 	--benchmark --model_type gptq  --input_len 1024 --max_new_tokens 128 --batch_size 1

OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=5 python quant_llama.py /data/scratch/llama-7b-hf c4 \
	--wbits 4 --true-sequential --act-order --groupsize 128 --load ./llama7b-4bit-128g-cai-nao.pt\
	--benchmark --model_type cai  --input_len 1024 --max_new_tokens 128 --batch_size 1

# OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=5 python quant_llama.py /data/scratch/llama-7b-hf c4 \
# 	--wbits 4 --true-sequential --act-order --groupsize 128 --load /llama7b-4bit-128g-gptq-nao.pt \
# 	--benchmark --model_type gptq --input_len 1024 --max_new_tokens 128 --batch_size 1

# OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=4 python quant_llama.py /data/scratch/llama-13b-hf c4 \
# 	--wbits 4 --true-sequential --act-order --groupsize 128  \
# 	--benchmark --model_type torch --input_len 1024 --max_new_tokens 128 --batch_size 1
