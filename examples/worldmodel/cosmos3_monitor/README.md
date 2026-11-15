# Cosmos 3 training monitor

A dashboard over the JSONL telemetry that `examples/worldmodel/cosmos3/train_sft.py` and
`examples/worldmodel/cosmos3_harness/train.py` already write. It turns a directory of run folders into a run
list, live charts, a configuration viewer, a node/card topology, a checkpoint ledger and a dashboard-only alert
list. The training-side additions are a 2 s heartbeat, one aggregated record per log step, a checkpoint ledger
after each save, terminal/session files at exit, and — behind `--graceful-stop`, on by default — a control
watcher that lets the dashboard stop a run after its current step and start it again from its latest
generation (see [Stop and resume](#stop-and-resume)).

## Why not a tracking SaaS

The cluster is isolated and runs Ascend NPUs; a hosted tracker is not reachable, and a self-hosted SwanLab or
Grafana stack would add a service to operate without covering the checkpoint ledger or the rank→card topology.
The dashboard is therefore a small FastAPI app plus a vendored Apache ECharts 6.1.0 bundle (Apache-2.0;
version, upstream URL and SHA-256 in `static/vendor/PROVENANCE.md`), fully offline. Its information architecture follows the mature parts of W&B/SwanLab (run list, config, comparison),
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
| `--graceful-stop` / `--no-graceful-stop` | honour `<run>/control/request.json` (default on; costs one 4-byte all-reduce per step) |
| `--stop-poll-every` | agree a stop every N steps (default 1) |
| `--control-poll-seconds` | how often the watcher re-reads the request file (default 1 s) |

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
- **Overview** — KPI tiles for iteration, epoch, batch, loss, step time, throughput, MFU, HBM, ranks and checkpoints; elapsed / ETA / total / finish-time line; loss/step-time charts; recent events; recent checkpoints; a compact run-info card. ETA uses the recent throughput (samples/s) and recent batch size, and total time is elapsed + ETA.
- **Charts** — loss, step-time breakdown, per-stage MFU (forward / backward / VAE), total + per-node throughput, rank × step throughput heatmap, MFU, LR, grad norm, memory, rank skew, batch/tokens and epoch. Step axes are integer-only; an elapsed-time axis is available. Toolbar: step/time axis, range, EMA smoothing, event/checkpoint annotations, log-y and zoom reset. Charts share zoom and crosshair; clicking a point opens the step detail.
- **Step detail** — opened from a chart point, a Records row or the iteration KPI. Shows iteration, epoch, global batch, micro-batch × accumulation, total and per-stage time/share/MFU/FLOPs, per-node and per-rank throughput, loss, optimizer and memory for that step, with previous/next navigation.
- **Records** — one row per aggregated log step (iteration, epoch, batch, loss, LR, grad, time, throughput, HBM, skew) with CSV export.
- **Config** — the full manifest as a searchable tree; copy/download; diff against another run.
- **Topology** — node × card grid with rank, phase, HBM and temperature; parallel group table.
- **Checkpoints** — step, kind, start/end, duration, size, files, path, async, source (ledger or filesystem),
  plus size/duration charts.
- **Events & Alerts** — training anomaly events and derived monitor alerts, with acknowledgement.
- **System** — per-host CPU/memory/disk/`/dev/shm`, per-card HBM/temperature/power, process table, history.
- **Logs** — tail of the path recorded in the manifest, with a path-traversal guard.
- **Compare** — select runs with the checkboxes on the Runs page; overlay loss and step time, compare KPI tables.

## Stop and resume

The run header shows two buttons, and the Overview has a `续训与停止` card with the channel state, the latest
generation and the command to run by hand.

**Stop** asks the loop to finish the step it is in, write one more resumable generation for exactly that
iteration and exit as `stopped` instead of `failed`. The request is a file, so it reaches a container the
dashboard cannot signal; the training process reads it in a background thread and only ever consults a boolean.
Every rank must leave on the same iteration — one rank leaving alone would pair its checkpoint all-gather with
another rank's gradient all-reduce and hang the job — so `boundary()` runs **one four-byte all-reduce per step**
whose maximum is "some rank has seen a request". That is the only cost, about 0.1 ms on a multi-second step;
`--no-graceful-stop` removes it for a benchmark.

Requests carry `run_started_at`, copied out of the run manifest. A new attempt in the same directory rewrites the
manifest, so a request aimed at the previous attempt cannot stop the resumed one, and no clock is compared across
machines.

**Resume** needs something that can start a process. Start the server with the command that launched the run,
with `{checkpoint}` and `{run_dir}` placeholders:

```bash
python -m examples.worldmodel.cosmos3_monitor serve --roots /mnt/disk0/runs --port 18790 \
    --launch-template 'cd /mnt/disk0/repo && bash train.sh --out {run_dir} --resume {checkpoint}'
```

The substituted values are shell-quoted, the child is detached (`start_new_session`) so it outlives the
dashboard, its output goes to `<run>/control/launch-<timestamp>.log`, and the dashboard remembers the pid and
exit code in memory. Without `--launch-template` the buttons still record the request and the card shows the
`--resume <root>` argument to add to the original command by hand — nothing is guessed.

The template starts **one** process on the host that runs the server. A multi-node run needs a template that
starts every node (the cluster launcher, or an `ssh` loop), because the dashboard has no opinion about
`torchrun --nnodes`. The server has no authentication: keep it on loopback and use the SSH tunnel.

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

The monitor writes only under its `--state-dir` (default `~/.cache/cosmos3_monitor`) unless a control action is
used, which writes `control/request.json` (and, with `--launch-template`, `control/launch-*.log`) inside the run.
`serve --no-control` removes even that.

## Interaction notes

- The topbar pause button freezes live updates; the SSE stream and the 5 s detail refresh resume together.
- `Pause` does not stop the sidebar's run discovery; it only stops the selected run's view from moving.
- Charts preserve the user's zoom across refreshes; a tab switch starts from the full range.
- The step detail's stage times are per-stage maxima across ranks, so their sum can be slightly larger than the
  maximum end-to-end step. Stage and total MFU come from the training record (`perf/mfu_forward`,
  `perf/mfu_backward`, `perf/mfu_vae`, `perf/mfu`): the trainer computed them with the checkpoint geometry and
  checkpointing mode it actually ran, so the dashboard has no second FLOPs accounting.  The SFT entry in #6553
  writes those fields; a run recorded before that change shows `—` rather than a guessed value.

## No training impact

- The heartbeat is a background thread writing a ~1 KB JSON every 2 s; `update()` only touches memory.
- `aggregated.jsonl` is written at the same log steps as the console/W&B sinks.
- The checkpoint ledger appends two small lines per save and reads size from the safetensors index or
  `complete.json`; it never walks a checkpoint tree in the hot path.
- The control watcher is a thread doing one `stat` and at most one small read per second; the training thread
  reads a boolean. Its one collective — the four-byte stop agreement — is the single per-step addition, and
  `--no-graceful-stop` removes it. No other per-step collective or device read is added.
- The collector is a separate process with a 5 s interval, timeouts and `nice 10`.
- The server only tails files; a slow filesystem makes a snapshot stale, not the training slower.

## SSH mirror (node-local metrics, all ranks)

When `--metrics-dir` is on a node-local disk (the validation cluster and the 6-node SFT benchmark do this),
rank 0's directory contains only its own eight `rank-*.jsonl`/`heartbeat-*` files.  Run the mirror from a host
that can SSH to every node; it polls the nodes in parallel and merges their disjoint files:

```bash
python -m examples.worldmodel.cosmos3_monitor mirror \
    --cache-root ~/.cache/cosmos3_monitor/mirror \
    --nodes node1,node5,node6,node7,node8,node9 \
    --run-id cosmos3_6node_1115 \
    --out /mnt/disk0/jiangzs/nzf/bench_online/runs/cosmos3_6node_1115 \
    --metrics-dir /mnt/disk0/jiangzs/nzf/bench_online/runs/cosmos3_6node_1115/metrics \
    --control-dir /mnt/disk0/jiangzs/nzf/bench_online/runs/cosmos3_6node_1115/control \
    --agent-path /mnt/disk0/jiangzs/nzf/c3agent/remote_agent.py \
    --remote-prefix 'docker exec nzf-cai-dev' --interval 5
python -m examples.worldmodel.cosmos3_monitor serve \
    --roots ~/.cache/cosmos3_monitor/mirror --port 18790
```

- Every node contributes its own `rank-*.jsonl`, `heartbeat-*.json`, `terminal-*.json` and
  `system/node-<host>.jsonl`; `run.json`, `summary.json`, `aggregated.jsonl`, `checkpoints.jsonl` and the log are
  taken from the first node only.  The Ranks KPI and topology then show all 48 ranks and six nodes, and a node
  with no heartbeat is visible as `0/8 live` instead of silently disappearing.
- `--remote-prefix` runs the standard-library agent inside a container (`docker exec ...`) when the host user
  cannot read the root-owned run directory; `--agent-path` points at `remote_agent.py` on each node (it has no
  dependencies and can be copied to a shared path).
- The first poll copies at most `--max-initial-bytes` (default 2 MiB) of each rank/system file; later polls send
  only appends.  `--once` performs a single poll, useful to validate the setup.
- File content is written only when it changes, and a heartbeat's remote age is preserved on the local mtime.
  A dead training therefore looks stale even while the mirror keeps polling.  For the same reason the mirror's
  own state file lives outside `metrics/`.
- The cache is keyed by run id and byte offsets are persisted in `mirror-state.json`, so a monitor restart does
  not lose history.  Point `serve --roots <cache-root>` at it.
- Rank 0's `aggregated.jsonl` sees only the ranks on its own node.  When that record is partial, the headline
  batch, throughput, MFU and ETA fall back to every mirrored `rank-*.jsonl`: FLOPs are summed and MFU is the
  mean of the values each rank's trainer recorded.  The step drawer marks the correction and uses the rank
  records for stage times, FLOPs and stage MFU.
- A partial step is corrected in place rather than left at `arrived_ranks x samples_per_rank`: while the mirror
  is still collecting, `partial_samples` is scaled by `expected_world / arrived_ranks` when that ratio matches
  `samples_per_rank x grad_accum` (the Records table marks it `estimated`).  Once every rank file arrives, the
  same chart point and record are replaced with measured samples, tokens, stage times and skew.

## Tests

```bash
python -m pytest -q examples/worldmodel/cosmos3_monitor/tests \
    examples/worldmodel/cosmos3_harness/tests/test_training_hooks.py \
    examples/worldmodel/cosmos3_harness/tests/test_training_metrics.py \
    examples/worldmodel/cosmos3_harness/tests/test_run_control.py
```

The core tests need neither an accelerator nor FastAPI; `tests/test_api_control.py` skips itself when FastAPI or
httpx is missing, so a workstation venv with `fastapi` exercises the control endpoints through `TestClient`.

## Limitations

- Alerts are dashboard-only; there is no webhook or email by default. `alerts.py` has one narrow interface to
  add later.
- AICore% is not collected; HBM, temperature, power and the process table are the reliable signals.
- The server is single-worker and in-memory. It rebuilds its index by scanning `--roots` on startup.
- The built-in resume launcher starts one process on the server's host. Multi-node runs need a
  `--launch-template` that starts every node, and a run served from an SSH mirror (whose cache is a copy, not the
  directory the training writes) or from a read-only mount offers no buttons at all.
- The control channel is for one run directory. Two runs sharing a directory (a resume that keeps the same
  `--out`) share the channel; the `run_started_at` stamp keeps a stale request from affecting the new attempt.
- Python 3.9+ is supported (the workstation's system Python); the training container uses 3.12.
