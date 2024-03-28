SAVE_DIR=""


BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples
SAVE_DIR=$BASE_DIR/temp/benchmark

rm -rf $SAVE_DIR

python $EXAMPLES_DIR/data_preparation_scripts/prepare_prompt_dataset.py --data_input_dirs "/home/yeanbang/data/dataset/sft_data/alpaca/data_preprocessed/train" \
    --conversation_template_config ./Opt.json \
    --tokenizer_dir  "facebook/opt-125m" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --num_samples_per_datafile 30
