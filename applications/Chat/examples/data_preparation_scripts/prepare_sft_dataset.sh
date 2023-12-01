# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_sft_data_llama/cache
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_sft_data_llama/jsonl
# rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_sft_data_llama/arrow

# python prepare_sft_dataset.py --data_input_dirs /mnt/tos/lcxyc/experiments/experiments5/tokenized_sft_data/jsonl \
#     --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
#     --data_cache_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_sft_data_llama/cache \
#     --data_jsonl_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_sft_data_llama/jsonl \
#     --data_arrow_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_sft_data_llama/arrow \
#     --num_spliced_dataset_bins 1 \
#     --num_samples_per_datafile 500


rm -rf /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/cache
rm -rf /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/jsonl
rm -rf /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow

python prepare_sft_dataset.py --data_input_dirs /mnt/tos/lcxyc/experiments/experiments5/tokenized_sft_data/jsonl \
    --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
    --data_cache_dir /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/cache \
    --data_jsonl_output_dir /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/jsonl \
    --data_arrow_output_dir /home/lcyab/data/data_rlhf/tokenized_sft_data_llama/arrow \
