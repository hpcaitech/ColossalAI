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
python ${BASE}/train_dummy.py --strategy naive --num_episodes 1 \
                              --max_timesteps 2 --update_timesteps 2 \
                              --max_epochs 1 --train_batch_size 2 --lora_rank 4

torchrun --standalone --nproc_per_node=2 ${BASE}/train_dummy.py \
         --strategy colossalai_gemini --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2\
         --pretrain 'facebook/opt-350m' --model opt --lora_rank 4\
         --save_path ${BASE}/actor_checkpoint_dummy.pt
python ${BASE}/inference.py --model_path ${BASE}/actor_checkpoint_dummy.pt --pretrain 'facebook/opt-350m' --model opt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_dummy.py \
         --strategy ddp --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2\
         --pretrain 'facebook/opt-350m' --model opt --lora_rank 4\
         --save_path ${BASE}/actor_checkpoint_dummy.pt
python ${BASE}/inference.py --model_path ${BASE}/actor_checkpoint_dummy.pt --pretrain 'facebook/opt-350m' --model opt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_dummy.py \
         --strategy colossalai_zero2 --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2\
         --pretrain 'gpt2' --model gpt2 --lora_rank 4\
         --save_path ${BASE}/actor_checkpoint_dummy.pt
python ${BASE}/inference.py --model_path ${BASE}/actor_checkpoint_dummy.pt --pretrain 'gpt2' --model gpt2

rm -rf ${BASE}/actor_checkpoint_dummy.pt

# train prompts
python ${BASE}/train_prompts.py $PROMPT_PATH --strategy naive --num_episodes 1 \
                                             --max_timesteps 2 --update_timesteps 2 \
                                             --max_epochs 1 --train_batch_size 2 --lora_rank 4

torchrun --standalone --nproc_per_node=2 ${BASE}/train_prompts.py $PROMPT_PATH \
         --strategy colossalai_zero2 --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2\
         --pretrain 'facebook/opt-350m' --model opt --lora_rank 4\
         --save_path ${BASE}/actor_checkpoint_prompts.pt
python ${BASE}/inference.py --model_path ${BASE}/actor_checkpoint_prompts.pt --pretrain 'facebook/opt-350m' --model opt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_prompts.py $PROMPT_PATH \
         --strategy ddp --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2\
         --pretrain 'gpt2' --model gpt2 --lora_rank 4\
         --save_path ${BASE}/actor_checkpoint_prompts.pt
python ${BASE}/inference.py --model_path ${BASE}/actor_checkpoint_prompts.pt --pretrain 'gpt2' --model gpt2

torchrun --standalone --nproc_per_node=2 ${BASE}/train_prompts.py $PROMPT_PATH \
         --strategy colossalai_gemini --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2\
         --pretrain 'gpt2' --model gpt2 --lora_rank 4\
         --save_path ${BASE}/actor_checkpoint_prompts.pt
python ${BASE}/inference.py --model_path ${BASE}/actor_checkpoint_prompts.pt --pretrain 'gpt2' --model gpt2

rm -rf ${BASE}/actor_checkpoint_prompts.pt
