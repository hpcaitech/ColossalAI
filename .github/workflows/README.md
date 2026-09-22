# CI/CD workflows

## Overview

Colossal-AI uses GitHub Actions for pull-request checks, GPU regressions,
documentation, releases, and repository maintenance. This document describes
the workflow files that are currently present in this directory.

A pull request that changes `version.txt` is treated as a release pull request
by the release workflows.

## Three-layer CI rollout

The replacement test pipeline is being qualified in three resource tiers. The
legacy build, example, and compatibility workflows remain enabled while the new
tiers are validated. Retiring a legacy workflow is a separate change that must
also migrate required checks, notifications, and repository rules.

| Layer | Workflow file | Trigger | Coverage and resources |
| --- | --- | --- | --- |
| 1 | `ci_layer_1_pr.yml` | Every pull request to `main` | Formatting and genuinely CPU-only unit tests on GitHub-hosted runners. |
| 2 | `ci_layer_2_gpu_pr.yml` | Trusted same-repository pull requests and manual dispatch | Core batches 2 and 5 on the dedicated `colossalai-e1-h20` runner. |
| 3 | `ci_layer_3_full.yml` | Weekly schedule and manual dispatch | The complete GPU batch set, including four- and eight-GPU coverage. |
| Monitor | `ci_gpu_idle_monitor.yml` | Hourly schedule and manual dispatch | Probes the authorized H20 pool; runs layer 2 with at least two idle GPUs and layer 3 with all eight idle GPUs. |
| Shared | `ci_reusable_gpu_batches.yml` | Called by layers 2 and 3 | Fetches the pinned source, runs selected batches, uploads reports, and writes the run summary. |

The execution flow is:

```text
layer trigger
    -> reusable GPU workflow
        -> run_gpu_batch.py
            -> run_colossalai_batch.sh
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
both of its batches can be admitted, and skips the full layer 3 matrix unless
all eight GPUs are idle. The normal per-batch selector checks occupancy again
immediately before every launch, so a stale hourly probe still fails closed.

### Rollout configuration

1. Keep the `colossalai-e1-h20` runner label restricted to this repository.
2. Set `COLOSSAL_GPU_POOL` when CI may use only a subset of physical GPU
   indices.
3. Optionally set `COLOSSAL_TORCH213_VENV`, `COLOSSAL_CUDA130_HOME`, and
   `COLOSSAL_GPU_LOCK`. The reusable workflow contains the current E1 paths as
   rollout defaults.
4. Make `Formatting` and `CPU unit tests` required only after layer 1 has passed
   repeatedly. Make layer 2 required only after GPU capacity and the trusted-PR
   policy have been agreed.
5. Never execute unreviewed fork pull-request code on the persistent
   self-hosted runner. External contributions must be reviewed and tested from
   a trusted ref.
6. Add another Torch/CUDA profile to layer 3 only after its isolated environment
   or pinned image has been provisioned.

## Active workflow inventory

### Tests and builds

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `build_on_pr.yml` | Pull request and branch lifecycle events | Legacy Colossal-AI build and unit-test workflow. |
| `build_on_schedule.yml` | Schedule or manual dispatch | Legacy scheduled full unit tests. |
| `example_check_on_pr.yml` | Pull request | Runs tests for examples affected by a pull request. |
| `example_check_on_schedule.yml` | Schedule or manual dispatch | Runs the scheduled example suite. |
| `example_check_on_dispatch.yml` | Manual dispatch | Runs one or more explicitly selected example directories. |
| `compatibility_test_on_pr.yml` | Changes to `version.txt` or `.compatibility` in a pull request | Tests the compatibility matrix used by a release pull request. |
| `compatibility_test_on_schedule.yml` | Schedule or manual dispatch | Runs the compatibility matrix from `.compatibility`. |
| `compatibility_test_on_dispatch.yml` | Manual dispatch | Runs the required Torch and CUDA version inputs. Neither input has an implicit `all` default. |
| `cuda_ext_check_before_merge.yml` | Release pull request or manual dispatch | Checks CUDA extension builds before release. |
| `report_test_coverage.yml` | Completion of the legacy PR build | Reports test coverage back to the pull request. |
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
| `e1-gpu-on-pr.yml` | Trusted internal pull request | Runs the existing E1 GPU regression while the layered pipeline is qualified. |

### Repository maintenance

| Workflow file | Trigger | Purpose |
| --- | --- | --- |
| `close_inactive.yml` | Schedule | Closes inactive issues according to the repository policy. |
| `translate_comment.yml` | Issue or issue-comment event | Translates issue content and comments. |
| `submodule.yml` | Schedule or manual dispatch | Detects and proposes submodule updates. |
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
