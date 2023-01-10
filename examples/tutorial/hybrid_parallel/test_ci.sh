#!/bin/bash

pip install -r requirements.txt

colossalai run --nproc_per_node 4 train.py --config config.py -s

ret=$?
if [ $ret -ne 0 ]; then
    "This example failed, please fix the bugs above."
    exit -1
fi