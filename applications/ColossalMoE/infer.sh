NUM_GPU=2
MODEL="/home/zhaoxuanlei/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/58301445dc1378584211722b7ebf8743ec4e192b"

# ep
torchrun --standalone --nproc_per_node $NUM_GPU infer.py \
    --model_name $MODEL \
    --plugin "ep" \
