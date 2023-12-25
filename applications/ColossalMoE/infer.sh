NUM_GPU=2
MODEL="8x7b"

# ep
torchrun --standalone --nproc_per_node $NUM_GPU infer.py \
    --model_name $MODEL \
    --plugin "ep" \
