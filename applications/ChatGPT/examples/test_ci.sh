#!/usr/bin/env bash

set -xue

if [ -z "$PROMPT_PATH" ]; then
    echo "Please set \$PROMPT_PATH to the path to prompts csv."
    exit 1
fi

BASE=$(realpath $(dirname $0))

export OMP_NUM_THREADS=8

# install requirements
pip install -r ${BASE}/requirements.txt

# train dummy
for strategy in ddp colossalai_gemini colossalai_zero2; do
    torchrun --standalone --nproc_per_node=2 ${BASE}/train_dummy.py --strategy ${strategy} --num_episodes 2 --max_timesteps 3 --update_timesteps 3 --max_epochs 3 --experience_batch_size 4 --train_batch_size 4
done

# train prompts
for strategy in ddp colossalai_gemini colossalai_zero2; do
    torchrun --standalone --nproc_per_node=2 ${BASE}/train_prompts.py $PROMPT_PATH --strategy ${strategy} --num_episodes 2 --max_timesteps 3 --update_timesteps 3 --max_epochs 3
done
