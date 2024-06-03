#!/bin/bash

set -xe
BASE=$(realpath $(dirname $0))

export RAY_NAMESPACE=admin
export DATA=/data/scratch/chatgpt/prompts.csv

# install requirements
pip install -r ${BASE}/requirements.txt

python ${BASE}/mmmt_prompt.py --prompt_path $DATA --num_makers 2 --num_trainers 2 --trainer_strategy colossalai_gemini --model opt --critic_model opt --pretrain facebook/opt-350m --critic_pretrain facebook/opt-125m --experience_batch_size 4 --train_batch_size 2
