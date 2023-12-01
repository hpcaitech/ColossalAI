rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/cache
rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/jsonl
rm -rf /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/arrow

python prepare_ptx_dataset.py --data_input_dirs /home/lcyab/data/pretrain_data/cleaned_wiki_en/preprocessed \
    --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
    --data_cache_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/cache \
    --data_jsonl_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/jsonl \
    --data_arrow_output_dir /home/lcyab/data/data_rlhf/test_tiny_data/tokenized_ptx_data_llama/arrow \
    --num_spliced_dataset_bins 1 \
    --num_samples_per_datafile 500


# rm -rf /home/lcyab/data/data_rlhf/tokenized_sft_data/cache
# rm -rf /home/lcyab/data/data_rlhf/tokenized_sft_data/jsonl
# rm -rf /home/lcyab/data/data_rlhf/tokenized_sft_data/arrow

# python prepare_sft_dataset.py --data_input_dirs /home/lcyab/data/data_rlhf/preprocessed_prompt \
#     --tokenizer_dir  "/home/lcyab/data/models/Sheared-LLaMA-1.3B" \
#     --data_cache_dir /home/lcyab/data/data_rlhf/tokenized_sft_data/cache \
#     --data_jsonl_output_dir /home/lcyab/data/data_rlhf/tokenized_sft_data/jsonl \
#     --data_arrow_output_dir /home/lcyab/data/data_rlhf/tokenized_sft_data/arrow \
#     --num_spliced_dataset_bins 1 \
#     --num_samples_per_datafile 500
