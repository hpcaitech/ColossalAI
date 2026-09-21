# E1 ColossalAI PR regression

The `ColossalAI GPU on internal PR` workflow runs two existing ColossalAI tests:

- `tests/test_booster/test_accelerator.py::test_accelerator`: verifies model placement on CPU and CUDA.
- `tests/test_booster/test_plugin/test_dp_plugin_base.py::test_dp_plugin_dataloader`: launches two NCCL workers through `colossalai.launch`, prepares a DPPlugin dataloader, and checks that ranks receive different data.

This is a small initial regression set, not the complete ColossalAI suite. The previous pinned-container infrastructure smoke test remains available through `pr_gpu.py --suite smoke`; the workflow now explicitly chooses `--suite colossalai`.

## Trigger and source

Every same-repository PR in `hpcaitech/ColossalAI` is eligible, regardless of author, head branch, base branch or changed paths. Opening, updating, reopening or marking a PR ready triggers the workflow. Draft PRs are also tested. External fork PRs are excluded from the self-hosted GPU job. Existing PRs need a subsequent event or an explicit re-run after rollout; merging the workflow does not retroactively start them all.

The workflow must be present in the PR merge snapshot. Publishing it to `main` covers normal PRs targeting `main`; independently maintained release branches may need the workflow backported. It does not automatically become a required merge check.

The runner label is `colossalai-e1-h20`. A label assigns a job to a runner; the workflow and the explicit `TESTS` list in `colossalai_suite.py` select the tests. Concurrency is grouped by PR number, so pending jobs for different PRs do not replace each other. The single runner executes jobs sequentially; multiple updates within one PR may replace its older pending run. Running jobs are not automatically cancelled.

Because the node's Git HTTPS endpoint was unreliable, the job downloads the official full source archive for `GITHUB_SHA` (the PR merge commit) and records its SHA256. Tests run from that fresh snapshot. An import-path assertion prevents accidentally testing the copy installed in the shared environment.

## Runtime and resources

The initial Python executable is `/mnt/beegfs/ColossalAI/wangzhijian/envs/colossalai-torch213/bin/python` (Torch 2.13/CUDA 13.0). The workflow does not install packages or modify this environment. It is a pre-existing runtime rather than an immutable CI image; dependency changes can require separate environment maintenance. Runtime versions are recorded in `colossalai/environment.json`.

A GPU-free preflight imports the actual source and collects the selected tests. The job then selects two idle GPU UUIDs using utilization, memory and compute processes. The suite acquires the same node-local advisory lock as the E1 smoke wrapper, rechecks occupancy, and exposes only those GPUs with `CUDA_VISIBLE_DEVICES`.

There is no reservation website access, waiting loop, or preemption. Fewer than two idle GPUs is a failure. The advisory lock coordinates E1 only; it cannot stop other users from starting work. These tests run as host processes in the trusted same-repository PR context, not in the smoke-test container. Do not enable untrusted fork code on this shared host.

## Results and limits

The parent bounds the suite to six minutes and terminates its process group on timeout/cancellation. The Actions job is bounded to 12 minutes. The pytest run must exit successfully and its JUnit XML must contain exactly the two selected passing test cases, with no failures, errors or skips. The infrastructure smoke marker cannot satisfy this check.

Evidence is retained under `/mnt/beegfs/ricardoo/ci/gpu-h20-5/test-results/pr-<run-id>-<attempt>/`: `result.json`, `test.log`, `colossalai/environment.json`, and `colossalai/junit.xml`. CPU-only preflight metadata is in the sibling `-preflight` directory. Actions logs and the job summary expose the status and tested commit. A busy-resource failure is not a passing regression run.
