export DATA=/data/scratch/imagenet/tf_records
export OMP_NUM_THREADS=4

# resume
# CUDA_VISIBLE_DEVICES=4,5,6,7 colossalai run \
# --nproc_per_node 4 train.py \
# --config configs/vit_1d_tp2.py \
# --resume_from checkpoint/epoch_10 \
# --master_port 29598 | tee ./out 2>&1

# train
CUDA_VISIBLE_DEVICES=4,5,6,7 colossalai run \
--nproc_per_node 4 train.py \
--config configs/vit_1d_tp2.py \
--master_port 29598 | tee ./out 2>&1
