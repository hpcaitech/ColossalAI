NUM_GPU=8
MODEL="/home/zhaoxuanlei/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/58301445dc1378584211722b7ebf8743ec4e192b"
SEQ_LENGTH=2048
BATCH_SIZE=1
LR=0.00001

# hybrid
# torchrun --standalone --nproc_per_node $NUM_GPU \
colossalai run --nproc_per_node $NUM_GPU --hostfile "hostfile" \
    train.py \
    --num_epoch 1 \
    --model_name $MODEL \
    --plugin "hybrid" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --zero_stage 1 \
    --pp_size 2 \
    --dp_size 1 \
    --ep_size 8 \
