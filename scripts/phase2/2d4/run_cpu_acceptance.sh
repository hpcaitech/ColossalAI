#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ColossalAI
D2D4_OFFLINE_FIXTURES=1 bash scripts/phase2/2d4/run_tests.sh tests/test_model_zoo/test_ouro.py -q
D2D4_OFFLINE_FIXTURES=0 bash scripts/phase2/2d4/run_tests.sh \
    tests/test_model_zoo/test_ouro.py \
    tests/test_shardformer/test_model/test_shard_remote_models.py \
    tests/test_shardformer/test_hf5_compat/test_d4_compat.py \
    tests/test_numerics/test_numerics_core.py -q
