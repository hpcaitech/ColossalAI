pip install -r requirements.txt

# inference
python infer.py --model "test"

# train
torchrun --standalone --nproc_per_node 4 train.py \
    --num_epoch 1 \
    --model_name "test" \
    --plugin zero2 \
    --batch_size 1
