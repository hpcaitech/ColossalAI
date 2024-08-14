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


NUM_RETRY=3
BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples
CONFIG_DIR=$BASE_DIR/config
TEMP_DIR=$BASE_DIR/temp
TEST_DIR=$BASE_DIR/tests
MODEL_SAVE_PATH=$TEMP_DIR/rlhf_models
MODELS_DIR=$TEMP_DIR/models_config
# Skip those tests due to CI tests timeout
MODELS=('llama')
ADVANCED_PLUGINS=('zero2' 'sp_split_gather' 'sp_ring' 'sp_all_to_all' 'tp_zero2' '3d' 'gemini' 'gemini_auto' 'zero2_cpu' 'pp' 'tp_pp')
PLUGINS=('zero2' '3d' 'gemini' 'gemini_auto' 'zero2_cpu')
LORA_RANK=('0')  # skip to reduce CI execution time, can pass all locally
LORA_CONFIG_ENABLE="--lora_config $BASE_DIR/examples/training_scripts/lora_config.json"

export OMP_NUM_THREADS=8

get_pretrain() {
    local model=$1
    if [[ $model == "llama" ]]; then
        echo "nickypro/tinyllama-110M"
    elif [[ $model == "opt" ]]; then
        echo "facebook/opt-125m"
    else
        echo "Unknown model $model"
        exit 1
    fi
}

get_tokenizer_dirs() {
    local model=$1
    if [[ $model == "llama" ]]; then
        echo "hf-internal-testing/llama-tokenizer"
    elif [[ $model == "opt" ]]; then
        echo "facebook/opt-125m"
    else
        echo "Unknown model $model"
        exit 1
    fi
}


get_conversation_template_config() {
    local model=$1
    if [[ $model == "llama" ]]; then
        echo "$TEST_DIR/llama.json"
    elif [[ $model == "opt" ]]; then
        echo "$TEST_DIR/opt.json"
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

SKIPPED_TESTS=(
    llama-3d-20 # 3d plugin doesn't support lora
    llama-gemini_auto-20  # gemini_auto plugin doesn't support lora
    llama-gemini-20 # gemini doesn't support lora
)
skip_eval=false
GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        for plugin in ${ADVANCED_PLUGINS[@]}; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            bs='2'
            pp='1'
            zero_stage='0'
            sp='1'
            sp_mode='split_gather'
            enable_sequence_parallelism=''
            if [[ $plugin == "zero2" ]]; then
                lora_config=$LORA_CONFIG_ENABLE
            else
                lora_config=""
            fi
            if [[ $plugin == "3d" ]]; then
                tp='2'
                bs='8'
            fi
            if [[ $plugin == "tp_zero2" ]]; then
                tp='2'
                bs='8'
                zero_stage='2'
                plugin='3d'
            fi
            if [[ $plugin == "tp_pp" ]]; then
                echo "Here"
                tp='2'
                bs='8'
                pp='2'
                plugin='3d'
                skip_eval=true
            fi
            if [[ $plugin == "pp" ]]; then
                bs='8'
                pp='2'
                plugin='3d'
                skip_eval=true
            fi
            if [[ $plugin == "sp_split_gather" ]]; then
                enable_sequence_parallelism='--enable_sequence_parallelism'
                sp_mode='split_gather'
                tp='2'
                sp='1'
                bs='8'
                plugin='3d'
            fi
            if [[ $plugin == "sp_ring" ]]; then
                enable_sequence_parallelism='--enable_sequence_parallelism'
                sp_mode='ring'
                tp='2'
                sp='1'
                bs='8'
                plugin='3d'
            fi
            if [[ $plugin == "sp_all_to_all" ]]; then
                enable_sequence_parallelism='--enable_sequence_parallelism'
                sp_mode='all_to_all'
                tp='1'
                sp='2'
                bs='8'
                plugin='3d'
            fi
            grad_accu='2'
            # Check if the plugin is either "gemini_auto" or "gemini" and set grad_accu to '1'
            if [[ $plugin == "gemini_auto" ]]; then
                grad_accu='1'
            fi

            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_sft/arrow/part-$split")
                done

                if [[ $skip_eval ]]; then
                    colossalai run --nproc_per_node 4 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_sft.py \
                        --pretrain $pretrain \
                        --tokenizer_dir $tokenizer_dir \
                        --dataset ${dataset[@]} \
                        --save_path $MODEL_SAVE_PATH \
                        --config_file $MODELS_DIR/config.jsonl \
                        $lora_config \
                        --plugin $plugin \
                        --batch_size $bs \
                        --max_epochs 1 \
                        --accumulation_steps $grad_accu \
                        --tp $tp \
                        --pp $pp \
                        --zero_stage $zero_stage \
                        --sp $sp \
                        --sp_mode $sp_mode \
                        $enable_sequence_parallelism \
                        --lr 2e-5 \
                        $grad_ckpt \
                        --max_len 400 \
                        --use_flash_attn
                else
                    colossalai run --nproc_per_node 4 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_sft.py \
                        --pretrain $pretrain \
                        --tokenizer_dir $tokenizer_dir \
                        --dataset ${dataset[@]} \
                        --eval_dataset ${dataset[@]} \
                        --save_path $MODEL_SAVE_PATH \
                        --config_file $MODELS_DIR/config.jsonl \
                        $lora_config \
                        --plugin $plugin \
                        --batch_size $bs \
                        --max_epochs 1 \
                        --accumulation_steps $grad_accu \
                        --tp $tp \
                        --pp $pp \
                        --zero_stage $zero_stage \
                        --sp $sp \
                        --sp_mode $sp_mode \
                        $enable_sequence_parallelism \
                        --lr 2e-5 \
                        $grad_ckpt \
                        --max_len 400 \
                        --use_flash_attn
                fi
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf ${MODEL_SAVE_PATH:?}/*
                    rm -rf ${MODELS_DIR:?}/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done

echo "[Test]: testing reward model ..."

SKIPPED_TESTS=(
    llama-3d-20 # 3d plugin doesn't support lora
    llama-gemini_auto-20  # gemini_auto plugin doesn't support lora
    llama-gemini-20 # gemini doesn't support lora
)

GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        for plugin in ${PLUGINS[@]}; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            bs='2'
            if [[ $plugin == "zero2" ]]; then
                lora_config=$LORA_CONFIG_ENABLE
            else
                lora_config=""
            fi
            if [[ $plugin == "3d" ]]; then
                tp='2'
                bs='8'
            fi
            grad_accu='2'
            # gemini_auto and gemini doesn't support gradient accumulation
            if [[ $plugin == "gemini_auto" ]]; then
                grad_accu='1'
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_preference/arrow/part-$split")
                done
                colossalai run --nproc_per_node 2 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_rm.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --eval_dataset ${dataset[@]} \
                    --save_dir $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    $lora_config \
                    --plugin $plugin \
                    --batch_size $bs \
                    --max_epochs 1 \
                    --accumulation_steps $grad_accu \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400 \
                    --use_flash_attn
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf ${MODEL_SAVE_PATH:?}/*
                    rm -rf ${MODELS_DIR:?}/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done


echo "[Test]: testing ppo ..."


SKIPPED_TESTS=(
    llama-3d # 3d plugin doesn't support lora
    llama-gemini # gemini doesn't support lora
)

GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        for plugin in ${PLUGINS[@]}; do
            if [[ $plugin == "gemini_auto" ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue # gemini_auto plugin doesn't support generation
            fi
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            bs='4'
            ebs='8'
            conversation_template=$(get_conversation_template_config $model)
            if [[ $plugin == "zero2" ]]; then
                lora_config=$LORA_CONFIG_ENABLE
            else
                lora_config=""
            fi
            if [[ $plugin == "3d" ]]; then
                tp='2'
                bs='16'
                ebs='32'
            fi
            grad_accu='2'
            # gemini_auto and gemini doesn't support gradient accumulation
            if [[ $plugin == "gemini_auto" ]]; then
                grad_accu='1'
            fi
            # gemini_auto and gemini doesn't support generation
            if [[ $plugin == "gemini_auto" ]]; then
                # gemini-auto doesn't support generation
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a prompt_dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    prompt_dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_prompt/arrow/part-$split")
                done
                declare -a ptx_dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    ptx_dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_sft/arrow/part-$split")
                done
                colossalai run --nproc_per_node 2 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_ppo.py \
                    --pretrain $pretrain \
                    --rm_pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --conversation_template_config $conversation_template \
                    --prompt_dataset ${prompt_dataset[@]} \
                    --ptx_dataset ${ptx_dataset[@]} \
                    --ptx_batch_size 1 \
                    --ptx_coef 0.2 \
                    --save_path $MODEL_SAVE_PATH \
                    $lora_config \
                    --plugin $plugin \
                    --num_episodes 5 \
                    --num_collect_steps 1 \
                    --num_update_steps 1 \
                    --experience_batch_size $ebs \
                    --train_batch_size $bs \
                    --accumulation_steps $grad_accu \
                    --lr 9e-6 \
                    --mixed_precision "bf16" \
                    --grad_clip 1.0 \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400 \
                    --max_seq_len 10 \
                    # --use_flash_attn
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf ${MODEL_SAVE_PATH:?}/*
                    rm -rf ${MODELS_DIR:?}/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done


echo "[Test]: testing DPO ..."

SKIPPED_TESTS=(
    llama-3d-20 # 3d plugin doesn't support lora
    llama-gemini_auto-20  # gemini_auto plugin doesn't support lora
    llama-gemini-20 # gemini doesn't support lora
)
GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        for plugin in ${PLUGINS[@]}; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            bs='2'
            if [[ $plugin == "3d" ]]; then
                tp='2'
                bs='8'
            fi
            if [[ $plugin == "zero2" ]]; then
                lora_config=$LORA_CONFIG_ENABLE
            else
                lora_config=""
            fi
            grad_accu='2'
            # gemini_auto and gemini doesn't support gradient accumulation
            if [[ $plugin == "gemini_auto" ]]; then
                grad_accu='1'
            fi
            # gemini_auto doesn't support generation
            # (need to calculate ref_model logits through forwarding in inference mode)
            if [[ $plugin == "gemini_auto" ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_preference/arrow/part-$split")
                done
                colossalai run --nproc_per_node 2 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_dpo.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --eval_dataset ${dataset[@]} \
                    --save_dir $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    $lora_config \
                    --plugin $plugin \
                    --batch_size $bs \
                    --max_epochs 1 \
                    --accumulation_steps $grad_accu \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400 \
                    --use_flash_attn
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf ${MODEL_SAVE_PATH:?}/*
                    rm -rf ${MODELS_DIR:?}/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done



echo "[Test]: testing ORPO ..."

SKIPPED_TESTS=(
    llama-3d-20 # 3d plugin doesn't support lora
    llama-gemini_auto-20  # gemini_auto plugin doesn't support lora
    llama-gemini-20 # gemini doesn't support lora
)
GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        for plugin in ${PLUGINS[@]}; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            bs='2'
            if [[ $plugin == "3d" ]]; then
                tp='2'
                bs='8'
            fi
            if [[ $plugin == "zero2" ]]; then
                lora_config=$LORA_CONFIG_ENABLE
            else
                lora_config=""
            fi
            grad_accu='2'
            # gemini_auto and gemini doesn't support gradient accumulation
            if [[ $plugin == "gemini_auto" ]]; then
                grad_accu='1'
            fi
            # gemini_auto doesn't support generation
            # (need to calculate ref_model logits through forwarding in inference mode)
            if [[ $plugin == "gemini_auto" ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_preference/arrow/part-$split")
                done
                colossalai run --nproc_per_node 2 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_orpo.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --eval_dataset ${dataset[@]} \
                    --save_dir $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    $lora_config \
                    --plugin $plugin \
                    --batch_size $bs \
                    --max_epochs 1 \
                    --accumulation_steps $grad_accu \
                    --tp $tp \
                    --lr 2e-5 \
                    $grad_ckpt \
                    --max_len 400 \
                    --use_flash_attn
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf ${MODEL_SAVE_PATH:?}/*
                    rm -rf ${MODELS_DIR:?}/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done



echo "[Test]: testing KTO ..."

SKIPPED_TESTS=(
    llama-3d-20 # 3d plugin doesn't support lora
    llama-gemini_auto-20  # gemini_auto plugin doesn't support lora
    llama-gemini-20 # gemini doesn't support lora
)
GRAD_CKPTS=('--grad_checkpoint')
for lora_rank in ${LORA_RANK[@]}; do
    for model in ${MODELS[@]}; do
        for plugin in ${PLUGINS[@]}; do
            if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin-$lora_rank " ]]; then
                echo "[Test]: Skipped $model-$plugin-$lora_rank"
                continue
            elif [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$plugin " ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            pretrain=$(get_pretrain $model)
            tokenizer_dir=$(get_tokenizer_dirs $model)
            grad_ckpt=$(random_choice "${GRAD_CKPTS[@]}")
            tp='1'
            bs='2'
            if [[ $plugin == "3d" ]]; then
                tp='2'
                bs='8'
            fi
            if [[ $plugin == "zero2" ]]; then
                lora_config=$LORA_CONFIG_ENABLE
            else
                lora_config=""
            fi
            grad_accu='2'
            # gemini_auto and gemini doesn't support gradient accumulation
            if [[ $plugin == "gemini_auto" ]]; then
                grad_accu='1'
            fi
            # gemini_auto doesn't support generation
            # (need to calculate ref_model logits through forwarding in inference mode)
            if [[ $plugin == "gemini_auto" ]]; then
                echo "[Test]: Skipped $model-$plugin"
                continue
            fi
            for i in $(seq $NUM_RETRY); do
                echo "[Test]: $model-$plugin-$lora_rank, attempt $i"
                declare -a dataset=()
                for split in $(seq -f "%05g" 0 0); do
                    dataset+=("$TEMP_DIR/rlhf_data/tokenized_${model}_kto/arrow/part-$split")
                done
                colossalai run --nproc_per_node 2 --master_port 31332 $EXAMPLES_DIR/training_scripts/train_kto.py \
                    --pretrain $pretrain \
                    --tokenizer_dir $tokenizer_dir \
                    --dataset ${dataset[@]} \
                    --eval_dataset ${dataset[@]} \
                    --save_dir $MODEL_SAVE_PATH \
                    --config_file $MODELS_DIR/config.jsonl \
                    $lora_config \
                    --plugin $plugin \
                    --batch_size $bs \
                    --max_epochs 1 \
                    --accumulation_steps $grad_accu \
                    --tp $tp \
                    --lr 2e-5 \
                    --auto_weight \
                    --desirable_weight 1.2 \
                    $grad_ckpt \
                    --max_len 400 \
                    --use_flash_attn
                passed=$?
                if [ $passed -eq 0 ]; then
                    rm -rf ${MODEL_SAVE_PATH:?}/*
                    rm -rf ${MODELS_DIR:?}/*
                    break
                fi
            done
            if [ $passed -ne 0 ]; then
                echo "[Test]: Failed $model-$plugin-$lora_rank"
                exit 1
            fi
        done
    done
done
