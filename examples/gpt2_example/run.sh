#!/usr/bin/env sh
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Ring
export NCCL_P2P_LEVEL=2
export NCCL_NET_GDR_LEVEL=5

export DATA=/path/to/data

torchrun --standalone --nproc_per_node=no_gpus train_gpt.py --config=configs/config_filename --from_torch