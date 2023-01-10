#!/bin/bash
set -euxo pipefail

# pip install -r requirements.txt

python -c "import titans"
# colossalai run --nproc_per_node 4 train.py --config config.py -s
