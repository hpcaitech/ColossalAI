NUM_GPU=2
# MODEL="mistralai/Mixtral-8x7B-v0.1"
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"

# ep
torchrun --standalone --nproc_per_node $NUM_GPU infer.py \
    --model_name $MODEL \
    --plugin "ep" \
