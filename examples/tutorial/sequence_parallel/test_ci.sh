#!/bin/bash
set -euxo pipefail

pip install -r requirements.txt

# run test
colossalai run --nproc_per_node 4 train.py
