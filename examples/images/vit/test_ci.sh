export OMP_NUM_THREADS=4

pip install -r requirements.txt

# train
colossalai run \
--nproc_per_node 4 train.py \
--config configs/vit_1d_tp2_ci.py \
--dummy_data
