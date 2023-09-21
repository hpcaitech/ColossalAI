torchrun --standalone --nproc_per_node 4 train.py \
    --model_name "base" \
    --plugin "hybrid" \
    --pp_size 2 \
    --dp_size 1 \
    --ep_size 2 \
    --use_kernel \
    --zero_stage 1 \
    --batch_size 4
