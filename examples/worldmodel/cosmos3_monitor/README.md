# Cosmos 3 training monitor

A read-only dashboard over the JSONL telemetry that `examples/worldmodel/cosmos3/train_sft.py` and
`examples/worldmodel/cosmos3_harness/train.py` already write. It turns a directory of run folders into a run
list, live charts, a configuration viewer, a node/card topology, a checkpoint ledger and a dashboard-only alert
list. Nothing here runs inside the training process; the only training-side additions are a 2 s heartbeat, one
aggregated record per log step, a checkpoint ledger after each save, and terminal/session files at exit.

## Why not a tracking SaaS

The cluster is isolated and runs Ascend NPUs; a hosted tracker is not reachable, and a self-hosted SwanLab or
Grafana stack would add a service to operate without covering the checkpoint ledger or the rank→card topology.
The dashboard is therefore a small FastAPI app plus a vendored Apache ECharts bundle (Apache-2.0), fully
offline. Its information architecture follows the mature parts of W&B/SwanLab (run list, config, comparison),
TensorBoard (step/time axes, smoothing), and Grafana (system panels, alert rules).

## Quick start

Inside the training container (it has `fastapi` and `uvicorn` from the repository requirements):

```bash
python -m examples.worldmodel.cosmos3_monitor serve \
    --roots /mnt/disk0/jiangzs/nzf/bench_online/runs \
    --host 127.0.0.1 --port 18790
```

Open it through an SSH tunnel from your workstation:

```bash
ssh -N -L 18790:127.0.0.1:18790 node6
# then http://127.0.0.1:18790
```

A workstation without the training stack can use a small virtual environment:

```bash
python3 -m venv ~/.cache/cosmos3-monitor-venv
~/.cache/cosmos3-monitor-venv/bin/pip install 'fastapi>=0.110' 'uvicorn==0.29.0'
~/.cache/cosmos3-monitor-venv/bin/python -m examples.worldmodel.cosmos3_monitor serve \
    --roots /path/to/runs --port 18790
```

`--roots` accepts several directories. A root may itself be a run. The scanner understands:

- `<run>/metrics/run.json` — the current SFT and harness layout;
- `<run>/run.json` — an older layout;
- `<run>/config.json` + `<run>/metrics/` — the validation cluster launcher and the SSH mirror.

## Training-side hooks

`train_sft.py` and the harness accept:

| flag | effect |
|---|---|
| `--run-id` / `--run-name` | override the generated identity and the display name |
| `--log-path` | path of the stdout log so the Logs tab can tail it |
| `--heartbeat-seconds` | heartbeat interval (default 2 s) |
| `--no-heartbeat` | disable the background heartbeat |
| `--no-checkpoint-ledger` | disable `checkpoints.jsonl` |
| `--no-aggregate-metrics` | disable `aggregated.jsonl` |

For a benchmark that must be free of extra processes, also start the collector with
`COSMOS3_MONITOR_COLLECT=0` or `--no-collect` (see below).

Chart resolution follows `--metrics-every` (default 10 optimizer steps). The liveness heartbeat is still 2 s and
the collector 5 s, but a new loss point appears every 10 steps. For a live chart on a long-step run, pass
`--metrics-every 1` or `2`; the cost is one aggregated JSONL line and one rank-0 aggregation per log step.
For exact global sample counts and throughput on a very short run, add `--collective-metrics`: it gathers the
record at each log step instead of reading rank files, so a point is never partial, at the cost of one
collective per log step. It is off by default; the dashboard marks `complete=false` points as partial and
does not compute throughput from a partial sum.

The per-node system collector is a separate process:

```bash
python -m examples.worldmodel.cosmos3_monitor collect \
    --run-dir /path/to/out --interval 5 --entry train_sft.py
```

It writes `<run>/metrics/system/node-<host>.jsonl` with CPU, memory, network, disk, `/dev/shm`, per-card HBM,
temperature, power and the training process table. AICore utilization is deliberately not collected: the
driver on this cluster reports 89–99% even when only dead processes remain.

`scripts/train_with_monitor.sh` starts the collector and then runs the training command unchanged:

```bash
bash examples/worldmodel/cosmos3/scripts/train_with_monitor.sh \
    --run-dir /path/to/out --interval 5 -- \
    python -m torch.distributed.run --nproc_per_node=8 train_sft.py --out /path/to/out ...
```

Run one collector per node. There is no shared state between them; the shared run directory is the join.

## What the pages show

- **Runs** — status, progress, last loss/step time/throughput/MFU/HBM, loss sparkline, alert count, age.
- **Overview** — KPI tiles for iteration, epoch, batch, loss, step time, throughput, MFU, HBM, ranks and checkpoints; loss/step-time charts; recent events; recent checkpoints; a compact run-info card.
- **Charts** — loss, step-time breakdown, per-stage MFU (forward / backward / VAE), throughput, MFU, LR, grad norm, memory, rank skew, batch/tokens and epoch. Step axes are integer-only; an elapsed-time axis is available. Toolbar: step/time axis, range, EMA smoothing, event/checkpoint annotations, log-y and zoom reset. Charts share zoom and crosshair; clicking a point opens the step detail.
- **Step detail** — opened from a chart point, a Records row or the iteration KPI. Shows iteration, epoch, global batch, micro-batch × accumulation, total and per-stage time/share/MFU/FLOPs, loss, optimizer and memory for that step, with previous/next navigation.
- **Records** — one row per aggregated log step (iteration, epoch, batch, loss, LR, grad, time, throughput, HBM, skew) with CSV export.
- **Config** — the full manifest as a searchable tree; copy/download; diff against another run.
- **Topology** — node × card grid with rank, phase, HBM and temperature; parallel group table.
- **Checkpoints** — step, kind, start/end, duration, size, files, path, async, source (ledger or filesystem),
  plus size/duration charts.
- **Events & Alerts** — training anomaly events and derived monitor alerts, with acknowledgement.
- **System** — per-host CPU/memory/disk/`/dev/shm`, per-card HBM/temperature/power, process table, history.
- **Logs** — tail of the path recorded in the manifest, with a path-traversal guard.
- **Compare** — select runs with the checkboxes on the Runs page; overlay loss and step time, compare KPI tables.

## Data contract

All files are additive; the per-rank JSONL remains the source of truth.

```
<run>/metrics/
  run.json                  # manifest (run identity, launcher env, topology, checkpoint schedule)
  summary.json              # final summary (steps, completed, duration, peak HBM, checkpoint totals)
  rank-000.jsonl ...        # existing per-rank records/events
  aggregated.jsonl          # rank 0, one record per log step, expected_world/complete/partial_samples
  heartbeat-000.json ...    # per rank, 2 s, atomic replace
  terminal-000.json ...     # per rank, written on exit
  checkpoints.jsonl         # begin/end records merged by checkpoint_id
  sessions.jsonl            # one line per launch/resume, one on exit
  system/node-<host>.jsonl  # per-node collector, 5 s
```

The monitor writes only under its `--state-dir` (default `~/.cache/cosmos3_monitor`), never inside a run.

## Interaction notes

- The topbar pause button freezes live updates; the SSE stream and the 5 s detail refresh resume together.
- `Pause` does not stop the sidebar's run discovery; it only stops the selected run's view from moving.
- Charts preserve the user's zoom across refreshes; a tab switch starts from the full range.
- The step detail's stage times are per-stage maxima across ranks, so their sum can be slightly larger than the
  maximum end-to-end step. Stage MFU assigns the activation recompute to backward; total step MFU uses 4x the
  forward FLOPs.

## No training impact

- The heartbeat is a background thread writing a ~1 KB JSON every 2 s; `update()` only touches memory.
- `aggregated.jsonl` is written at the same log steps as the console/W&B sinks.
- The checkpoint ledger appends two small lines per save and reads size from the safetensors index or
  `complete.json`; it never walks a checkpoint tree in the hot path.
- No new collective, synchronization or per-step device read is added.
- The collector is a separate process with a 5 s interval, timeouts and `nice 10`.
- The server only tails files; a slow filesystem makes a snapshot stale, not the training slower.

## SSH mirror (node-local metrics)

When `--metrics-dir` is on a node-local `/tmp` (the validation cluster did this deliberately), mirror it:

```bash
python -m examples.worldmodel.cosmos3_monitor mirror run-config.json \
    --cache-root ~/.cache/cosmos3_monitor/mirror \
    --remote-repo /mnt/disk0/jiangzs/nzf/ColossalAI \
    --nodes node5,node7,node8,node9 \
    --metrics-dir /tmp/run/metrics --control-dir /tmp/run/control --interval 5
```

The mirror materialises `<cache-root>/<run_id>/metrics/` with stable byte offsets in `mirror-state.json`, so a
monitor restart does not lose history. Point `serve --roots <cache-root>` at it.

## Tests

```bash
python -m pytest -q examples/worldmodel/cosmos3_monitor/tests \
    examples/worldmodel/cosmos3_harness/tests/test_training_hooks.py \
    examples/worldmodel/cosmos3_harness/tests/test_training_metrics.py
```

The core tests need neither an accelerator nor FastAPI; the API is exercised by running uvicorn in a fixture
only when FastAPI is available.

## Limitations

- Alerts are dashboard-only; there is no webhook or email by default. `alerts.py` has one narrow interface to
  add later.
- AICore% is not collected; HBM, temperature, power and the process table are the reliable signals.
- The server is single-worker and in-memory. It rebuilds its index by scanning `--roots` on startup.
- Python 3.9+ is supported (the workstation's system Python); the training container uses 3.12.
