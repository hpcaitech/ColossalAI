# CI/CD workflows

## Overview

Colossal-AI uses GitHub Actions for pull-request checks, GPU regressions,
documentation, releases, and repository maintenance. This document describes
the workflow files that are currently present in this directory.

A pull request that changes `version.txt` is treated as a release pull request
by the release workflows.

## Three-layer CI rollout

The replacement test pipeline is being qualified in three resource tiers. The
legacy example and compatibility workflows remain enabled for non-Draft PRs
while the new tiers are validated. Retiring any additional legacy workflow
must also account for required checks, notifications, and repository rules.

The initial `e1-gpu-on-pr.yml` host-venv workflow has been retired in favor of
the container-based Layer 2 workflow. Its accelerator test is covered by batch
2 and its two-rank DP dataloader test by batch 5. Do not restore the personal
Python-environment path as a fallback. The `colossalai-e1-h20` runner label and
the infrastructure-only `e1-runner-bootstrap.yml` probe are retained; retiring
a workflow does not unregister or reconfigure the runner.

Before merging this retirement, check branch protection and rulesets for the
old `gpu-smoke` check from `ColossalAI GPU on internal PR`. If it is required,
migrate that specific requirement to the qualified full Layer 2 check
`Torch 2.13 GPU core (batches 2 and 5) / run-batches`; do not substitute the
Draft-only smoke check or remove unrelated requirements. Removing a workflow
file does not update repository settings or cancel existing runs. Other PRs
will stop using E1 once their workflow snapshot includes this removal.

The legacy `Build on PR` workflow (`build_on_pr.yml`) and its dedicated
`report_test_coverage.yml` consumer have also been retired. CPU checks and GPU
regressions use the three-layer pipeline; container batch 3 retains the
`BUILD_EXT=1` AOT compilation check. The batch manifest, not the old broad
`pytest tests/` invocation, defines the replacement's coverage. Dependency,
toolchain, and kernel changes still need the relevant Layer 3 batches; a Layer
2 pass alone does not replace that validation.

Before merging, also check for required statuses named `Detect file change`
or `Build and Test Colossal-AI` from the old PR build and migrate their policy
to qualified layer checks. Deleting these files does not change branch
protection, cancel queued/running jobs, or unregister any runner. Old runs
retain their workflow snapshots. The old coverage-comment feature is removed,
not migrated: the new pytest reports do not imply code-coverage measurement.
Legacy scheduled builds, example/compatibility checks, and release workflows
are outside this retirement and remain enabled.

| Layer | Workflow file | Trigger | Coverage and resources |
| --- | --- | --- | --- |
| 1 | `ci_layer_1_pr.yml` | Every pull request to `main` | Formatting and genuinely CPU-only unit tests on GitHub-hosted runners. |
| Draft smoke | `ci_layer_2_gpu_pr.yml` | Trusted same-repository Draft pull requests | Only batch 4a, in a container on `colossalai-e1-h20`, using the separate smoke GPU pool. |
| 2 | `ci_layer_2_gpu_pr.yml` | Trusted same-repository Ready pull requests and manual dispatch | Core batches 2, 4a and 5 in containers on the `colossalai-e1-h20` runner. |
| 3 | `ci_layer_3_full.yml` | Weekly schedule and manual dispatch | The complete GPU batch set, including four- and eight-GPU coverage. |
| Monitor | `ci_gpu_idle_monitor.yml` | Hourly schedule and manual dispatch | Probes the authorized H20 pool; runs layer 2 with at least two idle GPUs and layer 3 with all eight idle GPUs. |
| Shared | `ci_reusable_gpu_batches.yml` | Called by layers 2 and 3 | Fetches the pinned source, runs selected batches, uploads reports, and writes the run summary. |

Layer 2 retains its historical check name, `Torch 2.13 GPU core (batches 2 and 5)`,
so existing required-check rules do not break; its actual batch list now also
includes `4a`. Rename that check only together with branch-protection settings.

### Draft PR container validation

A trusted same-repository Draft PR runs the separate check
`Torch 2.13 container smoke (batch 4a)`. It calls the same container executor as
the full GPU jobs, tests the PR merge SHA, uploads its own reports, and removes
only its own containers. Batch 4a requires one GPU, runs with `FAST_MODE=0`, and
is unaffected by `FAST_TEST`. The entire smoke job has a 15-minute limit,
including source download and waiting for the shared GPU lock.

Set `COLOSSAL_CI_SMOKE_GPU_POOL` to GPUs explicitly authorized for the smoke
test window (for example `3` for physical GPU 3 on the runner host). It is a
separate repository variable, **not** a replacement for `COLOSSAL_GPU_POOL`.
An unset/empty smoke pool fails before starting containers or allocating GPUs;
it never falls back to the main pool. Clear the smoke variable when temporary
authorization expires. Both profiles use the same `COLOSSAL_GPU_LOCK` and
`COLOSSAL_CI_IMAGE`; a smoke-only lock would not coordinate with scheduled CI.

When the PR becomes Ready, `ready_for_review` starts the full `2 4a 5` check
using the main pool. Converting it back to Draft cancels the previous Layer 2
run through the existing per-PR concurrency group and starts the smoke route.
The two jobs have different check names: a green smoke is not evidence that
the skipped full regression passed. Do not bypass required full checks to merge.
External forks and authors outside OWNER/MEMBER/COLLABORATOR do not run either
GPU route. Manual Layer 2 dispatch remains a full regression, not a smoke run.

Remaining legacy PR GPU/private-runner jobs are gated out for Draft PRs,
including compatibility matrices and CUDA extension builds. Their Ready-PR
behavior and existing explicit manual triggers are retained. These admission conditions do
not cancel legacy runs already started while a PR was Ready; inspect those runs
separately before assuming that converting a PR to Draft has freed every GPU.
Scheduled jobs are unchanged and may still hold the shared lock. CPU/lint jobs
remain available for Drafts and forks.

Execution path:

```text
layer trigger
    -> reusable GPU workflow
        -> verify the preloaded dependency image by immutable image ID
        -> run_gpu_batch.py (host GPU lock + idle-device checks)
            -> container_batch.py (one disposable container per batch)
                -> run_container_batch.sh (install this source SHA)
                    -> run_colossalai_batch.sh (existing test membership)
```

`.github/scripts/ci/run_colossalai_batch.sh` is the source of truth for batch
membership and per-batch GPU counts. Layer 2 selects the fast core subset;
layer 3 selects the full set. Keeping batch definitions in one place prevents
the two layers from silently diverging.

The `cpu` batch used by layer 1 is deliberately different from the historical
GPU batch 1. It hides CUDA and verifies that CUDA is unavailable before running
tests. GPU selection fails closed when device occupancy cannot be determined,
and a host lock prevents two CI jobs on this runner from selecting the same
devices. The host lock complements rather than replaces the team's external
reservation policy.

The hourly monitor is deliberately capacity-aware. It skips layer 2 unless
all of its batches can be admitted, and skips the full layer 3 matrix unless
all eight GPUs are idle. The normal per-batch selector checks occupancy again
immediately before every launch, so a stale hourly probe still fails closed.

### Rollout configuration

1. Keep the `colossalai-e1-h20` runner label restricted to this repository.
2. Set `COLOSSAL_GPU_POOL` to the **authorized** comma-separated physical GPU
   indices, and `COLOSSAL_GPU_LOCK` to the shared lock file used by every CI
   executor on this machine. Both variables are required; there is no personal
   directory default. The monitor and GPU jobs must use the same lock.
3. Build and validate the [dependency image](../../docker/torch213/README.md)
   on the runner's Docker daemon (or import it once with `docker load`). Set
   `COLOSSAL_CI_IMAGE` to its full local `sha256:<64 hex digits>` image ID.
   The runner needs Docker access, NVIDIA Container Toolkit, a compatible driver,
   and host Python 3.10+ with PyYAML. Publishing the image is **not required**.
   The Docker daemon/context used for provisioning and CI must match. CI no longer needs
   `COLOSSAL_TORCH213_VENV` or `COLOSSAL_CUDA130_HOME`. Layer 1 remains CPU-only.
4. Make `Formatting` and `CPU unit tests` required only after layer 1 has passed
   repeatedly. Make layer 2 required only after GPU capacity and the trusted-PR
   policy have been agreed.
5. Never execute unreviewed fork pull-request code on the persistent
   self-hosted runner. External contributions must be reviewed and tested from
   a trusted ref.
6. Add another Torch/CUDA profile to layer 3 only after its isolated environment
   or pinned image has been provisioned.

### Container execution and image rollout

The image holds Python dependencies and CUDA build tools, **not ColossalAI**.
The workflow fetches the exact source SHA and mounts it read-only. Each batch
copies it into its private container layer, creates a small writable venv that
references the image dependencies, and installs the checkout with `--no-deps
--no-build-isolation`. Batch 3 sets `BUILD_EXT=1` to check AOT compilation;
other batches use source-specific JIT kernels as needed. Compiler caches are
fresh per container. The initial rollout intentionally has no shared cache or
testmon incremental selection.

The container rejects a mismatch between the source's image recipe/locks and
the files embedded in the image. Updating a lock file alone cannot make CI
silently validate the old dependency environment.

The host selector passes **GPU UUIDs** to Docker, not `--gpus all`. Inside the
container, the allocated devices become logical `0,1,...`; host indices must
not be reused there. No personal venv, host CUDA toolkit, home directory,
credentials or Docker socket is mounted. Only the report bind is writable;
containers run as the runner UID/GID so artifact ownership stays correct.
Containers are not a security boundary for hostile pull requests: the host
scripts still have Docker access. Keep the trusted-ref policy and runner-group
restrictions.

| Repository variable | Default / purpose |
| --- | --- |
| `COLOSSAL_CI_IMAGE` | Required, full local `sha256:image-ID` (or an already-loaded registry digest). No tag, automatic pull or host-environment fallback. |
| `COLOSSAL_GPU_POOL` | Required, GPUs reserved/authorized for CI. Idle does not mean authorized. |
| `COLOSSAL_CI_SMOKE_GPU_POOL` | Required for Draft smoke only. Authorized candidate GPUs for its one-GPU batch 4a; never falls back to the main pool. |
| `COLOSSAL_GPU_LOCK` | Required, agreed shared host lock. Use the same path as legacy executors. |
| `COLOSSAL_CI_CPUS` | `8`, CPU quota per active container. |
| `COLOSSAL_CI_MEMORY` | `64g`, RAM limit; swap does not add to this limit. |
| `COLOSSAL_CI_SHM` | `8g`, private shared memory for multiprocessing/NCCL. |

Tune RAM/CPU limits with the administrator before the full eight-GPU matrix;
CPU-offload tests can need more RAM than the initial smoke profile. These limits
do not reserve GPU memory or cap image/layer disk usage. Docker build/load/run
write to daemon storage, so keeping the checkout and exported image `.tar` in a
personal directory is **not** equivalent to confining all writes there. Do not
move the shared daemon's data root or restart it to achieve personal isolation.

Bootstrap with **one PR**: first build the candidate from the local branch on
an approved builder; verify dependencies; build/load it on the test daemon;
qualify batches `4a`, `3`, `4b`, then `2 5` and cancellation cleanup on allocated
GPUs using `run_gpu_batch.py --image sha256:...`. None of these steps requires
opening or merging a PR, publishing the image, or upgrading the host venv.
Save the qualified image ID, source SHA and test reports, optionally archive the
image with `docker save`, then configure the runner/repository variables and
open the single combined PR. GitHub event triggering, permissions and artifact
upload still need verification on that PR; fix them on the same branch/PR.
Open it as a **Draft** first to qualify only the container smoke route. Configure
the smoke pool independently; do not shrink the main pool to one GPU, which
would prevent normal two-/four-/eight-GPU jobs from running. After the Draft
check succeeds, arrange resources for the full Layer 2 job and mark the same PR
Ready. The GPU profile is not fully qualified by one FP8 smoke test alone.
Merge only after reviewing failures and skipped coverage.
Do not merge the workflow switch before the image and runner are ready.
Subsequent code-only PRs reuse the image and install their own source. Rebuild
the image when the lock files, base/toolchain or Dockerfile change, not on every
PR or `version.txt` change. This CI image is separate from the user release image.

An image tar is a portable backup, not an executable container directory.
Load it once into the actual runner daemon when restoring or moving machines;
ordinary runs reuse the local image without repeatedly loading the tar. If
several machines share the runner label, preload the same image on each one
or restrict the job to the provisioned runner. No registry credentials are needed.

For rollback, select a previously **qualified** image ID compatible with
the source under test, or revert the executor switch. Do not silently fall back
to a developer's venv. The E1 host-venv PR workflow and legacy `Build on PR`
workflow are retired. Legacy scheduled builds, example/compatibility checks,
and the release pipeline are retained. The remaining legacy GPU/private-runner
PR jobs are gated out for Drafts.

`FAST_MODE=1` continues to reduce the model matrices of batches 2 and 5; batch
4a is unaffected. Layer 3 defaults to `FAST_MODE=0`, which unsets `FAST_TEST`.
Installing testmon does not enable selection: a later change must validate
coverage recording, image/batch-scoped persistent data, multiprocessing and
CUDA/C++ change invalidation before opting in. Core smoke and full regressions
must remain unconditional.

### Failure diagnosis and targeted validation

Layer 3 accepts an explicit `batches` input. After changing code, rerun the
affected batches (for example `10 8 9 9q`) with `fast_mode=false` before another
complete matrix. A capacity skip is not a successful GPU test, and an optional
dependency skip is not validation of that dependency's kernel. Keep skipped
test counts and reasons visible in the uploaded pytest reports.

Qwen2/Qwen3 combine pipeline and sequence parallelism: later pipeline stages
receive a local sequence slice, but both query and key mask lengths must match
the full sequence used by attention. A rank-local shape error can otherwise
look like a communication timeout if collective process-group destruction
blocks while the other ranks wait for messages. Avoid collective destruction
while unwinding an exception so the original worker error can reach pytest.

The batch wrapper uses `run_with_timeout.py` to supervise a dedicated test
process group and clean its workers before the logging `tee` waits for EOF.
The GPU runner also checks whether the batch process exited
while streaming logs; it must not wait indefinitely for EOF held open by
orphaned workers. In container mode it also removes the uniquely named container
while still holding the lock; both ownership labels and the exact name must
match. An `always()` step retries cleanup for this job's scope on cancellation.
It never runs `docker system prune`, daemon-wide cleanup or unrelated-process
termination. A runner crash, forced kill or unavailable Docker daemon can still
prevent cleanup: stop admitting jobs and have the administrator inspect the
reported container name/scope before returning the runner to service.

## Active workflow inventory

### Fork-safe pull request checks

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `fork_pr_checks.yml` | Every pull request to `main` | Runs repository lint, runner-policy checks, and selected CPU unit tests on isolated GitHub-hosted runners. |

The workflow uses the `pull_request` event, a read-only token, no secrets, and no
GPU or self-hosted runner. Results are reported on the upstream pull request's
Checks page, including pull requests from forks. GitHub may require a maintainer
to approve the first workflow run from an outside contributor.

This existing fork-safe gate remains enabled alongside layer 1 during rollout.
Consolidating their overlapping CPU and formatting checks and migrating required
check names is a separate change.

Existing jobs that execute unmerged code on GPU or private runners are limited
to branches in this repository. Because a `pull_request` workflow definition is
itself part of the untrusted pull request merge commit, that condition is defense
in depth rather than an administrative security boundary. Repository owners must
also enforce organization or enterprise Actions policies, or restrict self-hosted
runner groups to approved workflows, in GitHub settings.

### Tests and builds

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `build_on_schedule.yml` | Schedule or manual dispatch | Legacy scheduled full unit tests. |
| `example_check_on_pr.yml` | Pull request | Runs tests for examples affected by a pull request. |
| `example_check_on_schedule.yml` | Schedule or manual dispatch | Runs the scheduled example suite. |
| `example_check_on_dispatch.yml` | Manual dispatch | Runs one or more explicitly selected example directories. |
| `compatibility_test_on_pr.yml` | Changes to `version.txt` or `.compatibility` in a pull request | Tests the compatibility matrix used by a release pull request. |
| `compatibility_test_on_schedule.yml` | Schedule or manual dispatch | Runs the compatibility matrix from `.compatibility`. |
| `compatibility_test_on_dispatch.yml` | Manual dispatch | Runs the required Torch and CUDA version inputs. Neither input has an implicit `all` default. |
| `cuda_ext_check_before_merge.yml` | Release pull request or manual dispatch | Checks CUDA extension builds before release. |
| `requirements_compatibility.yml` | Scoped requirements changes or manual dispatch | Validates dependency installation and compatibility. |
| `run_chatgpt_examples.yml` | Pull request | Runs ChatGPT example checks when their scoped files change. |
| `run_chatgpt_unit_tests.yml` | Pull request | Runs ChatGPT unit tests when their scoped files change. |
| `run_colossalqa_unit_tests.yml` | Pull request | Runs ColossalQA unit tests when their scoped files change. |

### Documentation

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `doc_check_on_pr.yml` | Pull request | Checks documentation changes. |
| `doc_test_on_pr.yml` | Pull request | Builds and tests documentation for a pull request. |
| `doc_test_on_schedule.yml` | Schedule or manual dispatch | Runs scheduled documentation tests. |
| `doc_build_on_schedule_after_release.yml` | Schedule, release, or manual dispatch | Builds published documentation on schedule and after a release. |

### Releases

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `draft_github_release_post_after_merge.yml` | Release pull request, release event, or manual dispatch | Drafts the GitHub release notes. |
| `release_pypi_after_merge.yml` | Merged release pull request or manual dispatch | Publishes a release to PyPI. |
| `release_test_pypi_before_merge.yml` | Release pull request | Publishes a candidate build to TestPyPI. |
| `release_nightly_on_schedule.yml` | Schedule or manual dispatch | Publishes the nightly package. |
| `release_docker_after_publish.yml` | Published release or manual dispatch | Publishes the release Docker image after the GitHub release is published. |

### Runner qualification

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `e1-runner-bootstrap.yml` | Scoped push or manual dispatch | Qualifies the E1 self-hosted runner and its runtime. |

### Repository maintenance

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `close_inactive.yml` | Schedule | Closes inactive issues according to the repository policy. |
| `translate_comment.yml` | Issue or issue-comment event | Translates issue content and comments. |
| `report_leaderboard_to_lark.yml` | Schedule or manual dispatch | Sends the community activity report to Lark. |

## Compatibility configuration

`.compatibility` lists supported Torch/CUDA image tags, one per line. The
compatibility PR and scheduled workflows read this file to build their container
matrix. The manual compatibility workflow instead requires explicit comma-
separated Torch and CUDA version inputs.

## CUDA extension configuration

`.cuda_ext.json` controls which CUDA versions are checked by the extension build
workflow. Each entry defines the Torch installation command and CUDA image used
for the check.

```json
{
  "build": [
    {
      "torch_command": "",
      "cuda_image": ""
    }
  ]
}
```

## Manual runs

See the GitHub documentation for
[manually running a workflow](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow).
