#!/bin/bash
set -x

pip install -r requirements.txt

FAIL_LIMIT=3

for plugin in "torch_ddp" "torch_ddp_fp16" "gemini" "low_level_zero" "hybrid_parallel"; do
    for i in $(seq 1 $FAIL_LIMIT); do
        torchrun --standalone --nproc_per_node 4 finetune.py --target_f1 0.86 --plugin $plugin --model_type "bert" && break
        echo "Failed $i times"
        if [ $i -eq $FAIL_LIMIT ]; then
            echo "Failed $FAIL_LIMIT times, exiting"
            exit 1
        fi
    done
done
