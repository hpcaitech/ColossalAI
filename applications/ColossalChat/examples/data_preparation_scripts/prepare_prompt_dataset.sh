SAVE_DIR=""

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset.py --type prompt \
    --data_input_dirs /PATH/TO/PROMPT/DATASET \
    --conversation_template_config /PATH/TO/CHAT/TEMPLATE/CONFIG.json \
    --tokenizer_dir  "" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 1024
