rm -rf /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/cache
rm -rf /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/jsonl
rm -rf /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow

python prepare_prompt_dataset.py --data_input_dirs /home/lcyab/data/data_rlhf/preprocessed_prompt \
    --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
    --data_cache_dir /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/cache \
    --data_jsonl_output_dir /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/jsonl \
    --data_arrow_output_dir /home/lcyab/data/data_rlhf/tokenized_prompt_dataset_llama/arrow


# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/cache
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/jsonl
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/arrow

# python prepare_prompt_dataset.py --data_input_dirs /home/lcyab/data/data_rlhf/preprocessed_prompt \
#     --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
#     --data_cache_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/cache \
#     --data_jsonl_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/jsonl \
#     --data_arrow_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_prompt_data_llama/arrow \
#     --num_spliced_dataset_bins 1 \
#     --num_samples_per_datafile 500
