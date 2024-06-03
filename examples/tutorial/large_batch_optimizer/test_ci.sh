#!/bin/bash
set -euxo pipefail
echo "this test is outdated"

# pip install -r requirements.txt

# run test
# colossalai run --nproc_per_node 4 --master_port 29500 train.py --config config.py --optimizer lars
# colossalai run --nproc_per_node 4 --master_port 29501 train.py --config config.py --optimizer lamb
