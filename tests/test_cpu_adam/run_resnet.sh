export DATA=~/data
export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --standalone  \
    --nproc_per_node 4\
    --master_addr localhost \
    --master_port 29520 \
    resnet_cifar10.py \
    --log 'fp32.log' \
    $@
