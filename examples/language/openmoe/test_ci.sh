pip install -r requirements.txt

# inference
python infer.py --model "test"

# train
torchrun --standalone --nproc_per_node 2 train.py \
    --num_epoch 1 \
    --model_name "test" \
    --plugin zero2
