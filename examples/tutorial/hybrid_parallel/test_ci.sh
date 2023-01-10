#!/bin/bash

colossalai run --nproc_per_node 4 train.py --config config.py -s