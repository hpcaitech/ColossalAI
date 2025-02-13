SAVE_DIR="/home/yeanbang/data/competition_math/data/tokenized"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset.py --type prompt \
    --data_input_dirs /home/yeanbang/data/competition_math/data/raw \
    --conversation_template_config /home/yeanbang/ColossalAI/applications/ColossalChat/conversation_template/Qwen_Qwen2.5-7B-Instruct-GSM8K.json \
    --tokenizer_dir  "/mnt/jfs-hdd/share/models/Qwen2.5-7B-Instruct-1M" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 200
