rm -rf save_dir/cache
rm -rf save_dir/jsonl
rm -rf save_dir/arrow

python prepare_preference_dataset.py --data_input_dirs preference_data_dir \
    --tokenizer_dir  "pretrained/model/path" \
    --data_cache_dir save_dir/cache \
    --data_jsonl_output_dir save_dir/jsonl \
    --data_arrow_output_dir save_dir/arrow
