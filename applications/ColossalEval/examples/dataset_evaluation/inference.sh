torchrun --nproc_per_node=1 inference.py \
    --config "path to config file" \
    --load_dataset \
    --inference_save_path "path to save inference results"
