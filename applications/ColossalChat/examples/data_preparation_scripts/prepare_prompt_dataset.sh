SAVE_DIR="/home/yeanbang/data/competition_math/data/tokenized/prompt_new"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset.py --type prompt \
    --data_input_dirs /home/yeanbang/data/competition_math/data/raw/prompt \
    --conversation_template_config /home/yeanbang/ColossalAI/applications/ColossalChat/conversation_template/Qwen_Qwen2.5-7B-Instruct.json \
    --tokenizer_dir  "/mnt/jfs-hdd/share/models/Qwen2.5-3B" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 300
