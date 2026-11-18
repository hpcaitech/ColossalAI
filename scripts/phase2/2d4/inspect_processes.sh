#!/usr/bin/env bash
set -euo pipefail
docker top "${D2D4_CONTAINER:-wzj-colossalai-2d4-20261118}" -eo pid,ppid,etime,args
nvidia-smi --id=5,6 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
