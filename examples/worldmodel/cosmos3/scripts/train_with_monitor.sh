#!/bin/bash
# Start the per-node monitor sidecar, then run the training command unchanged.
#
#   bash train_with_monitor.sh --run-dir /path/to/out --interval 5 -- \
#       python -m torch.distributed.run --nproc_per_node=8 train_sft.py --out /path/to/out ...
#
# The sidecar writes only <run>/metrics/system/node-<host>.jsonl.  It never imports torch and never touches the
# training process.  Set COSMOS3_MONITOR_COLLECT=0 for a benchmark run that must not have any extra process.
set -uo pipefail

RUN_DIR=""
METRICS_DIR=""
INTERVAL=5
NODE_RANK=${NODE_RANK:-0}
ENTRY=train_sft.py
COLLECT=1
PYTHON_BIN=${PYTHON:-python3}

usage() {
  sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR=${2:?--run-dir needs a value}; shift 2 ;;
    --metrics-dir) METRICS_DIR=${2:?--metrics-dir needs a value}; shift 2 ;;
    --interval) INTERVAL=${2:?--interval needs a value}; shift 2 ;;
    --node-rank) NODE_RANK=${2:?--node-rank needs a value}; shift 2 ;;
    --entry) ENTRY=${2:?--entry needs a value}; shift 2 ;;
    --no-collect) COLLECT=0; shift ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) echo "train_with_monitor: unknown wrapper argument $1" >&2; usage ;;
  esac
done

if [[ -z "${RUN_DIR}" ]]; then
  echo "train_with_monitor: --run-dir is required" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "train_with_monitor: no training command after --" >&2
  exit 2
fi

COLLECTOR_PID=""
cleanup() {
  if [[ -n "${COLLECTOR_PID}" ]]; then
    kill "${COLLECTOR_PID}" 2>/dev/null || true
    wait "${COLLECTOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${COSMOS3_MONITOR_COLLECT:-1}" != "0" && "${COLLECT}" == "1" ]]; then
  COLLECTOR_ARGS=(--run-dir "${RUN_DIR}" --interval "${INTERVAL}" --node-rank "${NODE_RANK}" --entry "${ENTRY}" --nice 10)
  if [[ -n "${METRICS_DIR}" ]]; then
    COLLECTOR_ARGS+=(--metrics-dir "${METRICS_DIR}")
  fi
  "${PYTHON_BIN}" -m examples.worldmodel.cosmos3_monitor collect "${COLLECTOR_ARGS[@]}" &
  COLLECTOR_PID=$!
  echo "train_with_monitor: collector pid ${COLLECTOR_PID} -> ${RUN_DIR}/metrics/system" >&2
fi

"$@"
STATUS=$?
exit "${STATUS}"
