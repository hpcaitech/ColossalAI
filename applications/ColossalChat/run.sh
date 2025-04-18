export NCCL_BLOCKING_WAIT=1
CUDA_VISIBLE_DEVICES=4,5,6,7  python  rl_example.py --dataset /home/share/data/dataset/math_competition_train_short.jsonl --model /home/share/data/model/Qwen2.5-3B -t 1 -i 2 -b vllm
