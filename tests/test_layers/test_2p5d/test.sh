#!/bin/bash

python -m torch.distributed.launch test_2p5d.py --nproc_per_node 8  --host $HOST --port 29516 --world_size 8
