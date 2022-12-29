env OMP_NUM_THREADS=12 torchrun  --nproc_per_node 4  --master_port 29501  train.py --config palm_config.py
