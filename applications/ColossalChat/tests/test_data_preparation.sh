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

if [ -z "$PROMPT_DATASET" ]; then
    echo "Please set \$PROMPT_DATASET to the path to prompts."
    exit 1
fi

if [ -z "$PREFERENCE_DATASET" ]; then
    echo "Please set \$SFT_DATASET to the path to sft dataset."
    exit 1
fi

NUM_RETRY=3
BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
BASE_TEMP_DIR=$BASE_DIR/temp
TEST_DIR=$BASE_DIR/tests
EXAMPLES_DIR=$BASE_DIR/examples
DATA_SAVE_PATH=$BASE_TEMP_DIR/rlhf_data
CONFIG_DIR=$BASE_DIR/config
# Skip those tests due to CI tests timeout
MODELS=('llama')

if [ ! -d "$BASE_TEMP_DIR" ]; then
  mkdir "$BASE_TEMP_DIR"
  echo "Directory created successfully"
else
  echo "Directory already exists"
fi

if [ ! -d "$DATA_SAVE_PATH" ]; then
  mkdir "$DATA_SAVE_PATH"
  echo "Directory created successfully"
else
  echo "Directory already exists"
fi


export OMP_NUM_THREADS=8

# install requirements
pip install -r $EXAMPLES_DIR/requirements.txt

get_data_input_dirs() {
    local data_type=$1
    if [[ $data_type == "sft" ]]; then
        echo "$SFT_DATASET"
    elif [[ $data_type == "prompt" ]]; then
        echo "$PROMPT_DATASET"
    elif [[ $data_type == "preference" ]]; then
        echo "$PREFERENCE_DATASET"
    elif [[ $data_type == "kto" ]]; then
        echo "$KTO_DATASET"
    else
        echo "Unknown data type $data_type"
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

random_choice() {
    local arr=("$@")
    local len=${#arr[@]}
    local idx=$((RANDOM % len))
    echo ${arr[$idx]}
}

echo "Prepare dummy data for testing..."
python $TEST_DIR/generate_dummy_datasets_for_testing.py \
    --data_dir $(get_data_input_dirs sft) \
    --data_type "sft"

python $TEST_DIR/generate_dummy_datasets_for_testing.py \
    --data_dir $(get_data_input_dirs preference) \
    --data_type "preference"

python $TEST_DIR/generate_dummy_datasets_for_testing.py \
    --data_dir $(get_data_input_dirs prompt) \
    --data_type "prompt"

python $TEST_DIR/generate_dummy_datasets_for_testing.py \
    --data_dir $(get_data_input_dirs kto) \
    --data_type "kto"

echo "[Test]: testing prepare_preference_dataset.py ..."

# FIXME: This is a hack to skip tests that are not working
SKIPPED_TESTS=(
)

# test prepare_preference_dataset
for model in ${MODELS[@]}; do
    data_type="preference"
    if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$data_type " ]]; then
        echo "[Test]: Skipped $model-$data_type"
        continue
    fi
    cache_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/cache
    jsonl_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/jsonl
    arrow_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/arrow
    rm -rf $cache_dir
    rm -rf $jsonl_dir
    rm -rf $arrow_dir
    data_input_dirs=$(get_data_input_dirs $data_type)
    tokenizer_dir=$(get_tokenizer_dirs $model)
    conversation_template=$(get_conversation_template_config $model)
    for i in $(seq $NUM_RETRY); do
        echo "[Test]: $model-$data_type, attempt $i"
        python $EXAMPLES_DIR/data_preparation_scripts/prepare_dataset.py \
            --type preference \
            --data_input_dirs $data_input_dirs \
            --conversation_template_config $conversation_template \
            --tokenizer_dir $tokenizer_dir \
            --data_cache_dir $cache_dir \
            --data_jsonl_output_dir $jsonl_dir \
            --data_arrow_output_dir $arrow_dir \
            --max_length 400 \
            --num_samples_per_datafile 100 \
            --num_spliced_dataset_bins 1
        passed=$?
        if [ $passed -eq 0 ]; then
            break
        fi
    done
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed $model-$data_type"
        exit 1
    fi
done

echo "[Test]: testing prepare_sft_dataset.py ..."

# FIXME: This is a hack to skip tests that are not working
SKIPPED_TESTS=(
)

# test prepare_sft_dataset
for model in ${MODELS[@]}; do
    data_type="sft"
    if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$data_type " ]]; then
        echo "[Test]: Skipped $model-$data_type"
        continue
    fi
    cache_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/cache
    jsonl_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/jsonl
    arrow_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/arrow
    data_input_dirs=$(get_data_input_dirs $data_type)
    tokenizer_dir=$(get_tokenizer_dirs $model)
    conversation_template=$(get_conversation_template_config $model)
    for i in $(seq $NUM_RETRY); do
        rm -rf $cache_dir
        rm -rf $jsonl_dir
        rm -rf $arrow_dir
        echo "[Test]: $model-$data_type, attempt $i"
        python $EXAMPLES_DIR/data_preparation_scripts/prepare_dataset.py \
            --type sft \
            --data_input_dirs $data_input_dirs \
            --conversation_template_config $conversation_template \
            --tokenizer_dir $tokenizer_dir \
            --data_cache_dir $cache_dir \
            --data_jsonl_output_dir $jsonl_dir \
            --data_arrow_output_dir $arrow_dir \
            --max_length 400 \
            --num_samples_per_datafile 100 \
            --num_spliced_dataset_bins 1
        passed=$?
        if [ $passed -eq 0 ]; then
            break
        fi
    done
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed $model-$data_type"
        exit 1
    fi
done

echo "[Test]: testing prepare_prompt_dataset.py ..."

# FIXME: This is a hack to skip tests that are not working
SKIPPED_TESTS=(
)

# test prepare_prompt_dataset
for model in ${MODELS[@]}; do
    data_type="prompt"
    if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$data_type " ]]; then
        echo "[Test]: Skipped $model-$data_type"
        continue
    fi
    cache_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/cache
    jsonl_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/jsonl
    arrow_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/arrow
    data_input_dirs=$(get_data_input_dirs $data_type)
    tokenizer_dir=$(get_tokenizer_dirs $model)
    conversation_template=$(get_conversation_template_config $model)
    for i in $(seq $NUM_RETRY); do
        rm -rf $cache_dir
        rm -rf $jsonl_dir
        rm -rf $arrow_dir
        echo "[Test]: $model-$data_type, attempt $i"
        python $EXAMPLES_DIR/data_preparation_scripts/prepare_dataset.py \
            --type prompt \
            --data_input_dirs $data_input_dirs \
            --conversation_template_config $conversation_template \
            --tokenizer_dir $tokenizer_dir \
            --data_cache_dir $cache_dir \
            --data_jsonl_output_dir $jsonl_dir \
            --data_arrow_output_dir $arrow_dir \
            --max_length 400 \
            --num_samples_per_datafile 100 \
            --num_spliced_dataset_bins 1
        passed=$?
        if [ $passed -eq 0 ]; then
            break
        fi
    done
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed $model-$data_type"
        exit 1
    fi
done


echo "[Test]: testing prepare_kto_dataset.py ..."

# FIXME: This is a hack to skip tests that are not working
SKIPPED_TESTS=(
)

# test prepare_kto_dataset
for model in ${MODELS[@]}; do
    data_type="kto"
    if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$data_type " ]]; then
        echo "[Test]: Skipped $model-$data_type"
        continue
    fi
    cache_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/cache
    jsonl_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/jsonl
    arrow_dir=$DATA_SAVE_PATH/tokenized_${model}_${data_type}/arrow
    data_input_dirs=$(get_data_input_dirs $data_type)
    tokenizer_dir=$(get_tokenizer_dirs $model)
    conversation_template=$(get_conversation_template_config $model)
    for i in $(seq $NUM_RETRY); do
        rm -rf $cache_dir
        rm -rf $jsonl_dir
        rm -rf $arrow_dir
        echo "[Test]: $model-$data_type, attempt $i"
        python $EXAMPLES_DIR/data_preparation_scripts/prepare_dataset.py \
            --type kto \
            --data_input_dirs $data_input_dirs \
            --conversation_template_config $conversation_template \
            --tokenizer_dir $tokenizer_dir \
            --data_cache_dir $cache_dir \
            --data_jsonl_output_dir $jsonl_dir \
            --data_arrow_output_dir $arrow_dir \
            --max_length 400 \
            --num_samples_per_datafile 100 \
            --num_spliced_dataset_bins 1
        passed=$?
        if [ $passed -eq 0 ]; then
            break
        fi
    done
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed $model-$data_type"
        exit 1
    fi
done
