#!/usr/bin/env bash

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 4

set -xu

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

NUM_RETRY=3
BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples
MODELS_DIR=$BASE_DIR/examples/models_config

export OMP_NUM_THREADS=8

# install requirements
pip install -r $EXAMPLES_DIR/requirements.txt

python $EXAMPLES_DIR/download_model.py --model-dir $MODELS_DIR --config-only

get_pretrain() {
    local model=$1
    if [[ $model == "gpt2" ]]; then
        echo "gpt2"
    elif [[ $model == "bloom" ]]; then
        echo "bigscience/bloom-560m"
    elif [[ $model == "opt" ]]; then
        echo "facebook/opt-350m"
    else
        echo "Unknown model $model"
        exit 1
    fi
}

random_choice() {
    local arr=("$@")
    local len=${#arr[@]}
    local idx=$((RANDOM % len))
    echo ${arr[$idx]}
}

echo "[Test]: testing sft ..."

# FIXME: This is a hack to skip tests that are not working
#  - gpt2-colossalai_zero2-4: RuntimeError: torch.cat(): expected a non-empty list of Tensors, raised in prepare stage
#  - gpt2-ddp: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#  - llama-*: These tests can be passed locally, skipped for long execution time
SKIPPED_TESTS=(
    "gpt2-colossalai_zero2-4"
    "gpt2-ddp"
    "llama-ddp"
    "llama-colossalai_gemini"
    "llama-colossalai_zero2"
)

for lora_rank in '0' '4'; do
    for strategy in 'ddp' 'colossalai_gemini' 'colossalai_zero2'; do
        for model in 'gpt2' 'bloom' 'opt' 'llama'; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$strategy-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$strategy-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$strategy " ]]; then
                echo "[Test]: Skipped $model-$strategy"
                continue
            fi
            pretrain=$(get_pretrain $model)
            passed=0
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$strategy-$lora_rank, attempt $i"
                torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_sft.py \
                    --pretrain $pretrain --tokenizer $MODELS_DIR/$model \
                    --model $model --strategy $strategy --lora_rank $lora_rank \
                    --dataset $SFT_DATASET --max_datasets_size 32 \
                    --max_epochs 1 --batch_size 1 --accumulation_steps 1 \
                    --save_path $EXAMPLES_DIR/rlhf_models/sft_ckpt_${model}_${lora_rank}
                passed=$?
                if [ $passed -eq 0 ]; then
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$strategy-$lora_rank"
                exit 1
            fi
        done
    done
done

echo "[Test]: testing reward model ..."

# FIXME: This is a hack to skip tests that are not working
#  - gpt2-colossalai_zero2-4: RuntimeError: torch.cat(): expected a non-empty list of Tensors, raised in prepare stage
#  - gpt2-ddp: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#  - llama-*: These tests can be passed locally, skipped for long execution time
SKIPPED_TESTS=(
    "gpt2-colossalai_zero2-4"
    "gpt2-ddp"
    "llama-ddp"
    "llama-colossalai_gemini"
    "llama-colossalai_zero2"
)

LOSS_FNS=('log_sig' 'log_exp')
DATASETS=('Anthropic/hh-rlhf' 'Dahoas/rm-static')
for lora_rank in '0' '4'; do
    for strategy in 'ddp' 'colossalai_gemini' 'colossalai_zero2'; do
        for model in 'gpt2' 'bloom' 'opt' 'llama'; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$strategy-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$strategy-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$strategy " ]]; then
                echo "[Test]: Skipped $model-$strategy"
                continue
            fi
            pretrain=$(get_pretrain $model)
            loss_fn=$(random_choice "${LOSS_FNS[@]}")
            dataset=$(random_choice "${DATASETS[@]}")
            subset=$(if [[ $dataset == "Dahoas/rm-static" ]]; then echo "None"; else echo "harmless-base"; fi)
            passed=0
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$strategy-$lora_rank, attempt $i"
                torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_reward_model.py \
                    --pretrain $pretrain --tokenizer $MODELS_DIR/$model \
                    --model $model --strategy $strategy --lora_rank $lora_rank --loss_fn $loss_fn \
                    --dataset $dataset --subset $subset --test True \
                    --save_path $EXAMPLES_DIR/rlhf_models/rm_ckpt_${model}_${lora_rank}.pt
                passed=$?
                if [ $passed -eq 0 ]; then
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed to train reward model $model-$strategy-$lora_rank"
                exit 1
            fi
        done
    done
done

echo "[Test]: testing RLHF ..."

# FIXME: This is a hack to skip tests that are not working
#  - gpt2-ddp: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#  - llama-*: These tests can be passed locally, skipped for long execution time
SKIPPED_TESTS=(
    "gpt2-ddp"
    "llama-ddp"
    "llama-colossalai_gemini"
    "llama-colossalai_zero2"
)

for model in 'gpt2' 'bloom' 'opt' 'llama'; do
    for lora_rank in '0' '4'; do
        for strategy in 'ddp' 'colossalai_gemini' 'colossalai_zero2'; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$strategy " ]]; then
                echo "[Test]: Skipped $model-$strategy"
                continue
            fi
            pretrain=$(get_pretrain $model)
            passed=0
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$strategy-$lora_rank, attempt $i"
                torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_prompts.py \
                    --prompt_dataset $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
                    --strategy $strategy --model $model --tokenizer $MODELS_DIR/$model \
                    --num_episodes 1 --num_collect_steps 2 --num_update_steps 1 \
                    --train_batch_size 2 --lora_rank $lora_rank \
                    --pretrain $EXAMPLES_DIR/rlhf_models/sft_ckpt_${model}_${lora_rank} \
                    --rm_pretrain $pretrain --rm_path $EXAMPLES_DIR/rlhf_models/rm_ckpt_${model}_${lora_rank}.pt \
                    --save_path $EXAMPLES_DIR/rlhf_models/actor_checkpoint_prompts.pt
                passed=$?
                if [ $passed -eq 0 ]; then
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed to train RLHF $model-$strategy-$lora_rank"
                exit 1
            fi
        done
        rm -rf $EXAMPLES_DIR/rlhf_models/sft_ckpt_${model}_${lora_rank}
        rm $EXAMPLES_DIR/rlhf_models/rm_ckpt_${model}_${lora_rank}.pt
    done
done
rm $EXAMPLES_DIR/rlhf_models/actor_checkpoint_prompts.pt
