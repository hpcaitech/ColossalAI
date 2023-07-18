#!/usr/bin/env sh

root_path=$PWD
PY_FILE_PATH="$root_path/run_pretraining.py"

tensorboard_path="$root_path/tensorboard"
log_path="$root_path/exp_log"
ckpt_path="$root_path/ckpt"


mkdir -p $tensorboard_path
mkdir -p $log_path
mkdir -p $ckpt_path

export PYTHONPATH=$PWD

env OMP_NUM_THREADS=40 colossalai run --hostfile ./hostfile \
                --include GPU002,GPU003,GPU004,GPU007 \
                --nproc_per_node=8 \
                $PY_FILE_PATH \
                --master_addr GPU007 \
                --master_port 20024 \
                --lr 2.0e-4 \
                --train_micro_batch_size_per_gpu 190 \
                --eval_micro_batch_size_per_gpu 20 \
                --epoch 15 \
                --data_path_prefix /h5 \
                --eval_data_path_prefix /eval_h5 \
                --tokenizer_path /roberta \
                --bert_config /roberta/config.json \
                --tensorboard_path $tensorboard_path \
                --log_path $log_path \
                --ckpt_path $ckpt_path \
                --log_interval 50 \
                --mlm bert \
                --wandb \
                --checkpoint_activations \
