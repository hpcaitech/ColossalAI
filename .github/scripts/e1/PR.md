# E1 ColossalAI PR regression

The `E1 Immediate GPU on PR` workflow runs two existing ColossalAI tests:

- `tests/test_booster/test_accelerator.py::test_accelerator`: verifies model placement on CPU and CUDA.
- `tests/test_booster/test_plugin/test_dp_plugin_base.py::test_dp_plugin_dataloader`: launches two NCCL workers through `colossalai.launch`, prepares a DPPlugin dataloader, and checks that ranks receive different data.

This is a small initial regression set, not the complete ColossalAI suite. The previous pinned-container infrastructure smoke test remains available through `pr_gpu.py --suite smoke`; the workflow now explicitly chooses `--suite colossalai`.

## Trigger and source

Only same-repository PRs to `main`, from `ci/e1-runner-bootstrap` and authored by `richardoo-707`, are eligible. Opening, updating, or reopening the PR triggers the workflow when its E1 files, ColossalAI source, selected tests, test configuration or dependency declarations change. Draft PRs are supported. Nothing merges the PR automatically.

The runner label is `colossalai-e1-h20`. A label assigns a job to a runner; it does not select tests. The workflow and the explicit `TESTS` list in `colossalai_suite.py` select the tests. Supporting additional trusted branches requires a reviewed change to both the workflow condition and `validate_event` in `pr_gpu.py`.

Because the node's Git HTTPS endpoint was unreliable, the job downloads the official full source archive for `GITHUB_SHA` (the PR merge commit) and records its SHA256. Tests run from that fresh snapshot. An import-path assertion prevents accidentally testing the copy installed in the shared environment.

## Runtime and resources

The initial Python executable is `/mnt/beegfs/ColossalAI/wangzhijian/envs/colossalai-torch213/bin/python` (Torch 2.13/CUDA 13.0). The workflow does not install packages or modify this environment. It is a pre-existing runtime rather than an immutable CI image; dependency changes can require separate environment maintenance. Runtime versions are recorded in `colossalai/environment.json`.

A GPU-free preflight imports the actual source and collects the selected tests. The job then selects two idle GPU UUIDs using utilization, memory and compute processes. The suite acquires the same node-local advisory lock as the E1 smoke wrapper, rechecks occupancy, and exposes only those GPUs with `CUDA_VISIBLE_DEVICES`.

There is no reservation website access, waiting loop, or preemption. Fewer than two idle GPUs is a failure. The advisory lock coordinates E1 only; it cannot stop other users from starting work. These tests run as host processes in the trusted internal branch context, not in the smoke-test container. Do not enable untrusted fork code on this shared host.

## Results and limits

The parent bounds the suite to six minutes and terminates its process group on timeout/cancellation. The Actions job is bounded to 12 minutes. The pytest run must exit successfully and its JUnit XML must contain exactly the two selected passing test cases, with no failures, errors or skips. The infrastructure smoke marker cannot satisfy this check.

Evidence is retained under `/mnt/beegfs/ricardoo/ci/gpu-h20-5/test-results/pr-<run-id>-<attempt>/`: `result.json`, `test.log`, `colossalai/environment.json`, and `colossalai/junit.xml`. CPU-only preflight metadata is in the sibling `-preflight` directory. Actions logs and the job summary expose the status and tested commit. A busy-resource failure is not a passing regression run.
