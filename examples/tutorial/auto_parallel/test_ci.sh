#!/bin/bash
set -euxo pipefail

pip install titans
pip install pulp
conda install coin-or-cbc
colossalai run --nproc_per_node 4 auto_parallel_with_resnet.py -s
