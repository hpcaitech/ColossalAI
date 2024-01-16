SAVE_DIR="/home/yeanbang/data/experiments/sft/SlimOrca"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

# python prepare_sft_dataset.py --data_input_dirs sft_data_dir \
#     --tokenizer_dir  "pretrained/model/path" \
#     --data_cache_dir $SAVE_DIR/cache \
#     --data_jsonl_output_dir $SAVE_DIR/jsonl \
#     --data_arrow_output_dir $SAVE_DIR/arrow \


python prepare_sft_dataset.py --data_input_dirs "/home/yeanbang/data/dataset/sft_data/SlimOrca/data_preprocessed" \
    --conversation_template_config ../../config/conversation_template/Sheared-LLaMA.json \
    --tokenizer_dir  "princeton-nlp/Sheared-LLaMA-1.3B" \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \