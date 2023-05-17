set -xe

pip install -r requirements.txt

colossalai run --nproc_per_node 1 train.py --config config.py
