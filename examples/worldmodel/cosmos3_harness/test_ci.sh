#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${COLOSSAL_PYTHON:-python}"
RESULTS_ROOT="${COLOSSAL_RESULTS:-${TMPDIR:-/tmp}/cosmos3-example-ci}"

mkdir -p "${RESULTS_ROOT}"
export RESULTS_ROOT
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"

"${PYTHON}" examples/worldmodel/cosmos3_harness/train.py --dry-run \
    > "${RESULTS_ROOT}/train-dry-run.json"

"${PYTHON}" examples/worldmodel/cosmos3_harness/environment_report.py \
    --output "${RESULTS_ROOT}/environment.json" \
    > "${RESULTS_ROOT}/environment.log"

"${PYTHON}" - <<'PY' > "${RESULTS_ROOT}/heldout-manifest.json"
import json

from examples.worldmodel.cosmos3_harness.heldout import split_by_group, write_split_manifest

records = [
    {"sample_id": f"{group}-{index}", "trajectory_id": group}
    for group in range(20)
    for index in range(4)
]
train_ids, validation_ids, manifest = split_by_group(records, validation_fraction=0.2, seed=7)
assert not set(train_ids) & set(validation_ids)
write_split_manifest(manifest, __import__("os").environ["RESULTS_ROOT"] + "/heldout.json")
print(json.dumps({"train": len(train_ids), "validation": len(validation_ids)}))
PY

"${PYTHON}" examples/worldmodel/cosmos3_harness/benchmark.py \
    --backend naive --device cpu --samples 1 2 --ar-len 8 --dm-len 16 \
    --heads 4 --kv-heads 2 --head-dim 8 --dtype float32 \
    --warmup 1 --iterations 2 --repeats 1 \
    --output "${RESULTS_ROOT}/benchmark.json"

"${PYTHON}" examples/worldmodel/cosmos3_harness/evaluate_reasoner.py \
    --benchmarks MMBench-Dev HallusionBench \
    --manifest "${RESULTS_ROOT}/reasoner-manifest.json" \
    > "${RESULTS_ROOT}/reasoner-manifest.log"

CUDA_VISIBLE_DEVICES="" FAST_TEST=1 "${PYTHON}" -m pytest -q \
    examples/worldmodel/cosmos3_harness/tests/test_robolab.py \
    examples/worldmodel/cosmos3_harness/tests/test_training_metrics.py \
    examples/worldmodel/cosmos3_harness/tests/test_resume.py \
    tests/test_shardformer/test_layer/test_two_way_attn.py \
    tests/test_shardformer/test_model/test_cosmos_mot_packing.py \
    tests/test_shardformer/test_model/test_cosmos3_policy.py

printf 'Cosmos 3 example checks passed. Results: %s\n' "${RESULTS_ROOT}"
