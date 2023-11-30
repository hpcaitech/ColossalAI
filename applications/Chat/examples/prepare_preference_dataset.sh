# generate tinny test dataset
rm -rf /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/cache
rm -rf /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/jsonl
rm -rf /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow

python prepare_preference_dataset.py --data_input_dirs /home/lcyab/data/data_rlhf/preprcessed \
    --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
    --data_cache_dir /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/cache \
    --data_jsonl_output_dir /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/jsonl \
    --data_arrow_output_dir /home/lcyab/data/data_rlhf/tokenized_preference_data_llama/arrow

# generate tinny test dataset
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_preference_data_llama/cache
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_preference_data_llama/jsonl
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_preference_data_llama/arrow

# python prepare_preference_dataset.py --data_input_dirs /home/lcyab/data/data_rlhf/preprcessed \
#     --num_samples_per_datafile 120 \
#     --num_spliced_dataset_bins 1 \
#     --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
#     --data_cache_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_preference_data_llama/cache \
#     --data_jsonl_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_preference_data_llama/jsonl \
#     --data_arrow_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_preference_data_llama/arrow
