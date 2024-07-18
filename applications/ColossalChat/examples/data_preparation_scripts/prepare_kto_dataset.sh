SAVE_DIR="/home/nvme-share/home/yeanbang/data/experiments/kto"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

python prepare_dataset.py --type kto \
    --data_input_dirs /home/nvme-share/home/yeanbang/data/dataset/hh_rlhf/kto_format/data \
    --conversation_template_config /home/nvme-share/home/yeanbang/ColossalAI/applications/ColossalChat/config/conversation_template/llama2.json \
    --tokenizer_dir  "/home/nvme-share/share/models/Sheared-LLaMA-1.3B" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 1024
