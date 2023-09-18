set -xe
pip install -r requirements.txt

python infer.py --model "test"
torchrun --standalone --nproc_per_node 2 train.py --model_name "test" --batch_size 1 --num_epoch 1 --plugin zero2
torchrun --standalone --nproc_per_node 4 train.py --model_name "test" --batch_size 1 --num_epoch 1 --plugin hybrid --pp_size 2 --dp_size 1 --ep_size 2
