SAVE_DIR="/home/yeanbang/data/dataset/RLVR_dataset/rlvr_math_tulu/data/tokenized_CPM"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset.py --type prompt \
    --data_input_dirs /home/yeanbang/data/dataset/RLVR_dataset/rlvr_math_tulu/data/raw/zero_shot \
    --conversation_template_config /home/yeanbang/ColossalAI/applications/ColossalChat/conversation_template/MiniCPM-2b.json \
    --tokenizer_dir  "/home/yeanbang/data/model/MiniCPM-2B-128k" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 15000
