set -xe
pip install -r requirements.txt

python infer.py --model "test"
torchrun --standalone --nproc_per_node 2 train.py --model_name "test" --batch_size 1 --num_epoch 1 --plugin zero2
