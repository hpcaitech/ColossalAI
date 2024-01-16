SAVE_DIR="/home/yeanbang/data/experiments/dpo_ocra"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

# python prepare_preference_dataset.py --data_input_dirs preference_data_dir \
#     --tokenizer_dir  "pretrained/model/path" \
#     --data_cache_dir save_dir/cache \
#     --data_jsonl_output_dir save_dir/jsonl \
#     --data_arrow_output_dir save_dir/arrow

python prepare_preference_dataset.py --data_input_dirs "/home/yeanbang/data/dataset/rlhf_data/OcraDPO/data_preprocessed/train" \
    --conversation_template_config ../../config/conversation_template/Sheared-LLaMA.json \
    --tokenizer_dir  "princeton-nlp/Sheared-LLaMA-1.3B" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow
