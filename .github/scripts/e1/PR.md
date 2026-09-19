# E1 immediate GPU PR qualification

`E1 Immediate GPU on PR` verifies that a pull request dispatches a real two-GPU
test to the `colossalai-e1-h20` runner on `gpu-h20-5`.

This initial rollout only accepts same-repository pull requests to `main` from
`ci/e1-runner-bootstrap`, authored by `richardoo-707`. It reacts to opening,
updating and reopening that PR, and only to changes in the E1 scripts/workflow.
It deliberately supports a draft qualification PR. It does not merge the PR.

The workflow uses the ordinary `pull_request` event and a read-only repository
token. It does not use `pull_request_target`, PATs, booking credentials or website
reservations. Resource use is authorized manually for this development trial.
The branch/author restrictions are rollout limits, not a sandbox for untrusted
code; do not generalize this shared-host workflow to external contributions.

The job downloads the official GitHub source archive for the exact PR merge
commit (`GITHUB_SHA`) over HTTPS, extracts only the four E1 test scripts into a
fresh temporary directory, and records the archive SHA256. This avoids the
node's failing connection to the Git HTTPS endpoint on `github.com`; the
`codeload.github.com` endpoint is reachable. It does not use a mutable branch
snapshot or a pre-existing developer checkout, and it does not require a PAT.
This limited extraction is suitable for this infrastructure probe; project-wide
tests will need a full source checkout. Downloading is bounded to four minutes.

After source preparation, CPU-only unit tests validate the GPU selector and event guard.
The selector considers memory, utilization and running compute processes,
chooses two currently idle GPU UUIDs, and passes them to `run_gpu_smoke.sh`.
That wrapper checks occupancy again immediately before starting the container.
If fewer than two GPUs are idle, or a selected GPU becomes busy, the job fails
without preempting another user's task. There is no automatic reservation or
waiting queue in this immediate trial.

The container exposes only the chosen pair. A successful exit requires the
actual `E1_GPU_SMOKE_PASS` record with two NCCL workers, covering matrix
multiplication, all-reduce, DDP gradients and three analytical SGD updates.
The cached image is pinned to a digest and downloads no packages or models.
This is runner qualification, not ColossalAI project/version compatibility.

One E1 PR GPU job runs at a time. The job is bounded to 12 minutes and the GPU
test to six minutes with shorter container-level deadlines. Ordinary exits and
cancellation clean up the job's own container; host failure or SIGKILL can bypass
cleanup, so inspect the recorded container ID when recovering from those cases.

The Actions log and job summary record the PR head SHA, tested merge SHA,
runner, chosen GPU UUIDs, timestamps and the real result. Detailed wrapper logs
also remain in the runner's personal CI test-results directory. No GPU pass is
reported when selection, execution or numerical checks fail.

Before expanding beyond this PR trial, connect the real reservation system,
agree the eligible GPU pool, and review the workflow's trust and trigger policy.
