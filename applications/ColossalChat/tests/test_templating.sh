
BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
BASE_TEMP_DIR=$BASE_DIR/temp
EXAMPLES_DIR=$BASE_DIR/examples
TEST_DATA_DIR=$BASE_DIR/tests/test_data
DATA_SAVE_PATH=$BASE_TEMP_DIR/tests
CONFIG_DIR=$BASE_DIR/config

MODELS=("colossal-llama2" "llama2" "zephyr" "mistral" "chatGLM2" "Qwen" "Vicuna" "Yi")

get_pretrain() {
    local model=$1
    if [[ $model == "colossal-llama2" ]]; then
        echo "hpcai-tech/Colossal-LLaMA-2-7b-base"
    elif [[ $model == "llama2" ]]; then
        echo "hf-internal-testing/llama-tokenizer"
    elif [[ $model == "zephyr" ]]; then
        echo "HuggingFaceH4/zephyr-7b-beta"
    elif [[ $model == "mistral" ]]; then
        echo "mistralai/Mistral-7B-Instruct-v0.2"
    elif [[ $model == "chatGLM2" ]]; then
        echo "THUDM/chatglm2-6b"
    elif [[ $model == "Qwen" ]]; then
        echo "Qwen/Qwen-7B-Chat"
    elif [[ $model == "Vicuna" ]]; then
        echo "lmsys/vicuna-7b-v1.5"
    elif [[ $model == "Yi" ]]; then
        echo "01-ai/Yi-6B-Chat"
    else
        echo "Unknown model $model"
        exit 1
    fi
}

get_conversation_template_config() {
    local model=$1
    echo "$CONFIG_DIR/conversation_template/$model.json"
}

# Test SFT data Preparation
for model in ${MODELS[@]}; do
    echo "Testing SFT data templating for $model"
    SAVE_DIR=$DATA_SAVE_PATH/sft/$model
    rm -rf $SAVE_DIR/cache
    rm -rf $SAVE_DIR/jsonl
    rm -rf $SAVE_DIR/arrow
    pretrain=$(get_pretrain $model)
    conversation_template_config=$(get_conversation_template_config $model)
    python $EXAMPLES_DIR/data_preparation_scripts/prepare_dataset.py --type sft --data_input_dirs $TEST_DATA_DIR/sft \
        --tokenizer_dir $pretrain \
        --conversation_template_config $conversation_template_config \
        --data_cache_dir $SAVE_DIR/cache \
        --data_jsonl_output_dir $SAVE_DIR/jsonl \
        --data_arrow_output_dir $SAVE_DIR/arrow
    passed=$?
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed in the SFT data templating for $model"
        exit 1
    fi
    python $BASE_DIR/tests/verify_chat_data.py --data_source $TEST_DATA_DIR/sft/test_sft_data.jsonl \
        --to_verify_file $SAVE_DIR/jsonl/part-00005.jsonl --data_type sft
    passed=$?
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed in the SFT data templating test for $model"
        exit 1
    fi
done


# Test DPO/PPO data Preparation
for model in ${MODELS[@]}; do
    echo "Testing DPO/PPO data templating for $model"
    SAVE_DIR=$DATA_SAVE_PATH/dpo/$model
    rm -rf $SAVE_DIR/cache
    rm -rf $SAVE_DIR/jsonl
    rm -rf $SAVE_DIR/arrow
    pretrain=$(get_pretrain $model)
    conversation_template_config=$(get_conversation_template_config $model)
    python $EXAMPLES_DIR/data_preparation_scripts/prepare_dataset.py --type preference --data_input_dirs $TEST_DATA_DIR/dpo \
        --tokenizer_dir  $pretrain \
        --conversation_template_config $conversation_template_config \
        --data_cache_dir $SAVE_DIR/cache \
        --data_jsonl_output_dir $SAVE_DIR/jsonl \
        --data_arrow_output_dir $SAVE_DIR/arrow
    passed=$?
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed in the DPO data templating for $model"
        exit 1
    fi
    python $BASE_DIR/tests/verify_chat_data.py --data_source $TEST_DATA_DIR/dpo/test_dpo_data.jsonl \
        --to_verify_file $SAVE_DIR/jsonl/part-00005.jsonl --data_type dpo
    passed=$?
    if [ $passed -ne 0 ]; then
        echo "[Test]: Failed in the DPO data templating test for $model"
        exit 1
    fi
done
