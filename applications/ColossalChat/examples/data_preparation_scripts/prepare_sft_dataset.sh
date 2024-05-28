SAVE_DIR="/home/yeanbang/data/experiment/dataset/alpaca/test/Yi-1.5-6B"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

# python prepare_dataset.py --type sft \
#     --data_input_dirs /home/yeanbang/data/experiment/dataset/sft_data/test/sft-data \
#     --conversation_template_config /home/yeanbang/data/ColossalAI/applications/ColossalChat/config/conversation_template/THUDM_chatglm3-6b.json \
#     --tokenizer_dir  "/mnt/jfs-hdd/home/data/models/ChatGlm-6B" \
#     --data_cache_dir $SAVE_DIR/cache \
#     --data_jsonl_output_dir $SAVE_DIR/jsonl \
#     --data_arrow_output_dir $SAVE_DIR/arrow \


python prepare_dataset.py --type sft \
    --data_input_dirs /home/yeanbang/data/experiment/dataset/sft_data/test/sft-data \
    --conversation_template_config /home/yeanbang/data/ColossalAI/applications/ColossalChat/config/conversation_template/01-ai_Yi-1.5-9B-Chat.json \
    --tokenizer_dir  "/mnt/jfs-hdd/share/models/Yi-1.5-6B" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
