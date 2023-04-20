#!/usr/bin/env bash

set -xue

if [ -z "$SFT_DATASET" ]; then
    echo "Please set \$SFT_DATASET to the path to sft dataset."
    exit 1
fi

if [ -z "$PROMPT_PATH" ]; then
    echo "Please set \$PROMPT_PATH to the path to prompts csv."
    exit 1
fi

if [ -z "$PRETRAIN_DATASET" ]; then
    echo "Please set \$PRETRAIN_DATASET to the path to alpaca data."
    exit 1
fi

BASE=$(realpath $(dirname $0))

export OMP_NUM_THREADS=8

# install requirements
pip install -r ${BASE}/requirements.txt

wandb init -m offline

# train sft
torchrun --standalone --nproc_per_node=4 ${BASE}/train_sft.py --pretrain 'bigscience/bloom-560m' \
        --model 'bloom' --strategy colossalai_zero2 --lora_rank 4\
        --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
        --save_path ${BASE}/output

torchrun --standalone --nproc_per_node=4 ${BASE}/train_sft.py --pretrain 'gpt2' \
        --model 'gpt2' --strategy colossalai_zero2 \
        --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
        --save_path ${BASE}/output

torchrun --standalone --nproc_per_node=4 ${BASE}/train_sft.py --pretrain 'facebook/opt-350m' \
        --model 'opt' --strategy colossalai_zero2 --lora_rank 4\
        --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
        --save_path ${BASE}/output

torchrun --standalone --nproc_per_node=4 ${BASE}/train_sft.py --pretrain 'gpt2' \
        --model 'gpt2' --strategy ddp --lora_rank 4\
        --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
        --save_path ${BASE}/output

#torchrun --standalone --nproc_per_node=4 ${BASE}/train_sft.py --pretrain 'facebook/opt-350m' \
#        --model 'opt' --strategy naive \
#        --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
#        --save_path ${BASE}/output

rm -rf ${BASE}/output

# train rm
torchrun --standalone --nproc_per_node=2 ${BASE}/train_reward_model.py \
                            --pretrain 'facebook/opt-350m' --model 'opt' \
                            --strategy colossalai_zero2 --loss_fn 'log_sig'\
                            --dataset 'Anthropic/hh-rlhf' --subset 'harmless-base' \
                            --test True --lora_rank 4 \
                            --save_path ${BASE}/rm_ckpt_opt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_reward_model.py \
                            --pretrain 'gpt2' --model 'gpt2' \
                            --strategy colossalai_zero2 --loss_fn 'log_exp' \
                            --dataset 'Dahoas/rm-static' \
                            --test True  --lora_rank 4 \
                            --save_path ${BASE}/rm_ckpt_gpt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_reward_model.py \
                            --pretrain 'gpt2' --model 'gpt2' \
                            --strategy ddp --loss_fn 'log_exp' \
                            --dataset 'Dahoas/rm-static' \
                            --test True --lora_rank 4 \
                            --save_path ${BASE}/rm_ckpt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_reward_model.py \
                            --pretrain 'bigscience/bloom-560m' --model 'bloom' \
                            --strategy colossalai_zero2 --loss_fn 'log_sig' \
                            --dataset 'Anthropic/hh-rlhf' --subset 'harmless-base' \
                            --test True --lora_rank 4 \
                            --save_path ${BASE}/rm_ckpt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_reward_model.py \
                            --pretrain 'microsoft/deberta-v3-large' --model 'deberta' \
                            --strategy colossalai_zero2 --loss_fn 'log_sig' \
                            --dataset 'Anthropic/hh-rlhf' --subset 'harmless-base' \
                            --test True --lora_rank 4 \
                            --save_path ${BASE}/rm_ckpt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_reward_model.py \
                            --pretrain 'roberta-base' --model 'roberta' \
                            --strategy colossalai_zero2 --loss_fn 'log_exp'\
                            --dataset 'Anthropic/hh-rlhf' --subset 'harmless-base'\
                            --test True --lora_rank 4 \
                            --save_path ${BASE}/rm_ckpt.pt

rm -rf ${BASE}/rm_ckpt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_prompts.py --prompt_path $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
        --strategy colossalai_zero2 --num_episodes 1 --max_timesteps 2 \
        --update_timesteps 2 --max_epochs 1 --train_batch_size 2 \
        --pretrain 'facebook/opt-350m' --model opt \
        --rm_pretrain 'facebook/opt-350m' \
        --rm_path ${BASE}/rm_ckpt_opt.pt \
        --save_path ${BASE}/actor_checkpoint_prompts.pt
rm -rf ${BASE}/rm_ckpt_opt.pt

torchrun --standalone --nproc_per_node=2 ${BASE}/train_prompts.py --prompt_path $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
         --strategy colossalai_zero2 --num_episodes 1 --max_timesteps 2 \
         --update_timesteps 2 --max_epochs 1 --train_batch_size 2 \
         --pretrain 'gpt2' --model gpt2 \
         --rm_pretrain 'gpt2' \
         --rm_path ${BASE}/rm_ckpt_gpt.pt \
         --save_path ${BASE}/actor_checkpoint_prompts.pt
rm -rf ${BASE}/rm_ckpt_gpt.pt

rm -rf ${BASE}/actor_checkpoint_prompts.pt