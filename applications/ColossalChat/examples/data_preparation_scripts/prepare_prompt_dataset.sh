SAVE_DIR="/home/yeanbang/data/dataset/RLVR_dataset/gsm8k/data/tokenized"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset.py --type prompt \
    --data_input_dirs /home/yeanbang/data/dataset/RLVR_dataset/gsm8k/data/raw \
    --conversation_template_config /home/yeanbang/ColossalAI/applications/ColossalChat/conversation_template/MiniCPM-2b.json \
    --tokenizer_dir  "/home/yeanbang/data/model/MiniCPM-2B-128k" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 500
