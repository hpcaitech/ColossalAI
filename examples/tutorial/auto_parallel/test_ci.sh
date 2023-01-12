#!/bin/bash
set -euxo pipefail

conda init bash
conda env create -f environment.yaml
conda activate auto
cd ../../..
pip uninstall colossalai
pip install -v .
cd ./examples/tutorial/auto_parallel
colossalai run --nproc_per_node 4 auto_parallel_with_resnet.py -s
