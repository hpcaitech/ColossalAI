pip install -r requirements.txt

# inference
python infer.py --model "test"

# train
torchrun --standalone --nproc_per_node 4 train.py \
    --num_epoch 1 \
    --model_name "test" \
    --plugin zero2_ep \
    --batch_size 1

torchrun --standalone --nproc_per_node 4 train.py \
    --model_name "test" \
    --plugin "hybrid" \
     --num_epoch 1 \
    --pp_size 2 \
    --dp_size 1 \
    --ep_size 2 \
    --zero_stage 1 \
    --batch_size 1
