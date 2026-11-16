#!/bin/bash
# One short, deterministic Cosmos 3 SFT screen on 8 NPUs, for InfraRSI to search over.
#
# The knobs the search may move are environment variables with defaults at the bottom of this header. Everything
# that changes *what is computed* is pinned, not exposed:
#
#   --samples-per-rank  fixes the per-step workload; changing it changes the MFU denominator, not the efficiency
#   --seed / --max-episodes / --image-augmentation  fix the data stream. Augmentation draws come from each
#       DataLoader worker's RNG, so changing --num-workers changes the pixels and therefore the loss; num-workers
#       is pinned for the same reason.
#
# Usage: ./train.sh            (the search runs it as-is and rewrites the defaults when it proposes a change)
set -uo pipefail

LOCAL=/mnt/disk0/jiangzs/nzf
REPO=$LOCAL/ColossalAI
export PATH=$LOCAL/cai-npu/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-bond1}
export HCCL_CONNECT_TIMEOUT=600
export COLOSSALAI_EXTENSION_CACHE=$LOCAL/ext_cache
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TILELANG_CACHE_DIR=$LOCAL/tilelang_cache
mkdir -p "$COLOSSALAI_EXTENSION_CACHE"

# --- search knobs ---------------------------------------------------------------------------------------------
# 10 updates is the shortest run whose steady-state window is not drowned by the one-off kernel warmup.
STEPS=${STEPS:-10}
MAX_EPISODES=${MAX_EPISODES:-64}
GRAD_ACCUM=${GRAD_ACCUM:-1}
ZERO_BUCKET=${ZERO_BUCKET:-12}
VAE_BATCH=${VAE_BATCH:-8}
CKPT=${CKPT:-full}
FAST_VAE=${FAST_VAE:-1}
ATTN_BACKEND=${ATTN_BACKEND:-auto}
COMPILE_BLOCKS=${COMPILE_BLOCKS:-0}
VAE_COMPILE=${VAE_COMPILE:-none}
NUM_WORKERS=${NUM_WORKERS:-4}
# --------------------------------------------------------------------------------------------------------------

FAST_VAE_FLAG=""
[ "$FAST_VAE" = "1" ] && FAST_VAE_FLAG="--fast-vae-encoder"
COMPILE_FLAG=""
[ "$COMPILE_BLOCKS" = "1" ] && COMPILE_FLAG="--compile-blocks"

LOG=${LOG:-/tmp/infrarsi-screen.log}
cd "$REPO/examples/worldmodel/cosmos3" || exit 1

echo "=== CONFIG steps=$STEPS grad_accum=$GRAD_ACCUM zero_bucket=$ZERO_BUCKET vae_batch=$VAE_BATCH ckpt=$CKPT"
echo "=== CONFIG fast_vae=$FAST_VAE attn=$ATTN_BACKEND compile_blocks=$COMPILE_BLOCKS vae_compile=$VAE_COMPILE"
echo "=== CONFIG num_workers=$NUM_WORKERS omp_threads=$OMP_NUM_THREADS"

python -u -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  --master_addr=127.0.0.1 --master_port=${MASTER_PORT:-29613} \
  train_sft.py \
  --checkpoint /mnt/disk0/jiangzs/checkpoints/Cosmos3-Edge-Policy-DROID \
  --data /mnt/disk0/jiangzs/datasets/Cosmos3-DROID/success \
  --out /tmp/infrarsi-screen-run \
  --max-iter "$STEPS" --log-every 1 --metrics-every 1 --save-every 100000 \
  --samples-per-rank 32 --grad-accum "$GRAD_ACCUM" --hsdp-shard-size 8 \
  --num-workers "$NUM_WORKERS" --seed 42 --max-episodes "$MAX_EPISODES" \
  --image-augmentation torchvision \
  --zero-bucket-size-m "$ZERO_BUCKET" --vae-encode-batch "$VAE_BATCH" \
  --activation-checkpointing "$CKPT" --attn-backend "$ATTN_BACKEND" \
  --vae-compile "$VAE_COMPILE" $FAST_VAE_FLAG $COMPILE_FLAG "$@" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

# A one-screen summary, so the search reads numbers instead of re-deriving them from the per-step lines.
python - "$LOG" <<'PY'
import re, statistics, sys

lines = [l for l in open(sys.argv[1], errors="replace") if "[step " in l and "step " in l and "| samples" in l]
rows = []
for l in lines:
    m = re.search(r"step ([0-9.]+)s \(forward ([0-9.]+), backward ([0-9.]+), optim ([0-9.]+)\)", l)
    s = re.search(r"samples (\d+)", l)
    v = re.search(r"vae ([0-9.]+), wait ([0-9.]+)", l)
    mem = re.search(r"mem ([0-9.]+) GiB", l)
    loss = re.search(r"loss ([0-9.]+)", l)
    if m and s:
        rows.append(dict(step=float(m.group(1)), fwd=float(m.group(2)), bwd=float(m.group(3)),
                         optim=float(m.group(4)), samples=int(s.group(1)),
                         vae=float(v.group(1)) if v else 0.0, wait=float(v.group(2)) if v else 0.0,
                         mem=float(mem.group(1)) if mem else 0.0, loss=float(loss.group(1)) if loss else 0.0))

if not rows:
    print("=== SUMMARY status=no_steps_parsed")
    raise SystemExit(0)

# Step 1 carries the one-off kernel warmup and the cold data fill; it is reported separately, never averaged in.
steady = rows[1:] or rows
# Nominal per-update workload, not the count from the log: the reported "samples" column counts only the ranks that
# landed inside the metric window and varies 96..256 between updates whose wall time is identical, so deriving a
# rate from it would add noise the search cannot distinguish from a real change.
nominal_samples = 8 * 32
step_s = statistics.median(r["step"] for r in steady)
print("=== SUMMARY")
print(f"steps_parsed={len(rows)} steps_steady={len(steady)}")
print(f"samples_per_step={nominal_samples}")
print(f"steady_step_s={step_s:.3f}")
print(f"steady_fwd_s={statistics.median(r['fwd'] for r in steady):.3f}")
print(f"steady_bwd_s={statistics.median(r['bwd'] for r in steady):.3f}")
print(f"steady_vae_s={statistics.median(r['vae'] for r in steady):.3f}")
print(f"steady_wait_s={statistics.median(r['wait'] for r in steady):.3f}")
print(f"samples_per_s={nominal_samples / step_s:.2f}")
print(f"peak_hbm_gib={max(r['mem'] for r in rows):.2f}")
print(f"final_loss={rows[-1]['loss']:.4f}")
print(f"cold_step1_s={rows[0]['step']:.3f}")
PY

exit "$STATUS"
