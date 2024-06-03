# CI/CD

## Table of Contents

- [CI/CD](#cicd)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Workflows](#workflows)
    - [Code Style Check](#code-style-check)
    - [Unit Test](#unit-test)
    - [Example Test](#example-test)
      - [Example Test on Dispatch](#example-test-on-dispatch)
    - [Compatibility Test](#compatibility-test)
      - [Compatibility Test on Dispatch](#compatibility-test-on-dispatch)
    - [Release](#release)
    - [User Friendliness](#user-friendliness)
    - [Community](#community)
  - [Configuration](#configuration)
  - [Progress Log](#progress-log)

## Overview

Automation makes our development more efficient as the machine automatically run the pre-defined tasks for the contributors.
This saves a lot of manual work and allow the developer to fully focus on the features and bug fixes.
In Colossal-AI, we use [GitHub Actions](https://github.com/features/actions) to automate a wide range of workflows to ensure the robustness of the software.
In the section below, we will dive into the details of different workflows available.

## Workflows

Refer to this [documentation](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow) on how to manually trigger a workflow.
I will provide the details of each workflow below.

**A PR which changes the `version.txt` is considered as a release PR in the following context.**


### Code Style Check

| Workflow Name | File name         | Description                                                                                                    |
| ------------- | ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `post-commit` | `post_commit.yml` | This workflow runs pre-commit checks for changed files to achieve code style consistency after a PR is merged. |

### Unit Test

| Workflow Name          | File name                  | Description                                                                                                                                       |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Build on PR`          | `build_on_pr.yml`          | This workflow is triggered when a PR changes essential files and a branch is created/deleted. It will run all the unit tests in the repository with 4 GPUs. |
| `Build on Schedule`    | `build_on_schedule.yml`    | This workflow will run the unit tests everyday with 8 GPUs. The result is sent to Lark.                                                           |
| `Report test coverage` | `report_test_coverage.yml` | This PR will put up a comment to report the test coverage results when `Build` is done.                                                           |

To reduce the average time of the unit test on PR, `Build on PR` workflow manages testmon cache.

1. When creating a new branch, it copies `cache/main/.testmondata*` to `cache/<branch>/`.
2. When creating a new PR or change the base branch of a PR, it copies `cache/<base_ref>/.testmondata*` to `cache/_pull/<pr_number>/`.
3. When running unit tests for each PR, it restores testmon cache from `cache/_pull/<pr_number>/`. After the test, it stores the cache back to `cache/_pull/<pr_number>/`.
4. When a PR is closed, if it's merged, it copies `cache/_pull/<pr_number>/.testmondata*` to `cache/<base_ref>/`. Otherwise, it just removes `cache/_pull/<pr_number>`.
5. When a branch is deleted, it removes `cache/<ref>`.

### Example Test

| Workflow Name              | File name                       | Description                                                                    |
| -------------------------- | ------------------------------- | ------------------------------------------------------------------------------ |
| `Test example on PR`       | `example_check_on_pr.yml`       | The example will be automatically tested if its files are changed in the PR    |
| `Test example on Schedule` | `example_check_on_schedule.yml` | This workflow will test all examples every Sunday. The result is sent to Lark. |
| `Example Test on Dispatch` | `example_check_on_dispatch.yml` | Manually test a specified example.                                             |

#### Example Test on Dispatch

This workflow is triggered by manually dispatching the workflow. It has the following input parameters:
- `example_directory`: the example directory to test. Multiple directories are supported and must be separated by comma. For example, language/gpt, images/vit. Simply input language or simply gpt does not work.

### Compatibility Test

| Workflow Name                    | File name                            | Description                                                                                                          |
| -------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `Compatibility Test on PR`       | `compatibility_test_on_pr.yml`       | Check Colossal-AI's compatibility when `version.txt` is changed in a PR.                                              |
| `Compatibility Test on Schedule` | `compatibility_test_on_schedule.yml` | This workflow will check the compatibility of Colossal-AI against PyTorch specified in `.compatibility` every Sunday. |
| `Compatibility Test on Dispatch`  | `compatibility_test_on_dispatch.yml` | Test PyTorch Compatibility manually.                                                                                 |


#### Compatibility Test on Dispatch
This workflow is triggered by manually dispatching the workflow. It has the following input parameters:
- `torch version`:torch version to test against, multiple versions are supported but must be separated by comma. The default is value is all, which will test all available torch versions listed in this [repository](https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_build/torch_wheels).
- `cuda version`: cuda versions to test against, multiple versions are supported but must be separated by comma. The CUDA versions must be present in our [DockerHub repository](https://hub.docker.com/r/hpcaitech/cuda-conda).

> It only test the compatibility of the main branch


### Release

| Workflow Name                                   | File name                                   | Description                                                                                                   |
| ----------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Draft GitHub Release Post`                     | `draft_github_release_post_after_merge.yml` | Compose a GitHub release post draft based on the commit history when a release PR is merged.                  |
| `Publish to PyPI`                               | `release_pypi_after_merge.yml`              | Build and release the wheel to PyPI when a release PR is merged. The result is sent to Lark.                  |
| `Publish Nightly Version to PyPI`               | `release_nightly_on_schedule.yml`           | Build and release the nightly wheel to PyPI as `colossalai-nightly` every Sunday. The result is sent to Lark. |
| `Publish Docker Image to DockerHub after Merge` | `release_docker_after_merge.yml`            | Build and release the Docker image to DockerHub when a release PR is merged.  The result is sent to Lark.     |
| `Check CUDA Extension Build Before Merge`       | `cuda_ext_check_before_merge.yml`           | Build CUDA extensions with different CUDA versions when a release PR is created.                              |
| `Publish to Test-PyPI Before Merge`             | `release_test_pypi_before_merge.yml`        | Release to test-pypi to simulate user installation when a release PR is created.                              |


### User Friendliness

| Workflow Name           | File name               | Description                                                                                                                            |
| ----------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `issue-translate`       | `translate_comment.yml` | This workflow is triggered when a new issue comment is created. The comment will be translated into English if not written in English. |
| `Synchronize submodule` | `submodule.yml`         | This workflow will check if any git submodule is updated. If so, it will create a PR to update the submodule pointers.                 |
| `Close inactive issues` | `close_inactive.yml`    | This workflow will close issues which are stale for 14 days.                                                                           |

### Community

| Workflow Name                                | File name                        | Description                                                                      |
| -------------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| `Generate Community Report and Send to Lark` | `report_leaderboard_to_lark.yml` | Collect contribution and user engagement stats and share with Lark every Friday. |

## Configuration

This section lists the files used to configure the workflow.

1. `.compatibility`

This `.compatibility` file is to tell GitHub Actions which PyTorch and CUDA versions to test against. Each line in the file is in the format `${torch-version}-${cuda-version}`, which is a tag for Docker image. Thus, this tag must be present in the [docker registry](https://hub.docker.com/r/pytorch/conda-cuda) so as to perform the test.

2. `.cuda_ext.json`

This file controls which CUDA versions will be checked against CUDA extension built. You can add a new entry according to the json schema below to check the AOT build of PyTorch extensions before release.

```json
{
  "build": [
    {
      "torch_command": "",
      "cuda_image": ""
    },
  ]
}
```

## Progress Log

- [x] Code style check
  - [x] post-commit check
- [x] unit testing
  - [x] test on PR
  - [x] report test coverage
  - [x] regular test
- [x] release
  - [x] pypi release
  - [x] test-pypi simulation
  - [x] nightly build
  - [x] docker build
  - [x] draft release post
- [x] example check
  - [x] check on PR
  - [x] regular check
  - [x] manual dispatch
- [x] compatibility check
  - [x] check on PR
  - [x] manual dispatch
  - [x] auto test when release
- [x] community
  - [x] contribution report
  - [x] user engagement report
- [x] helpers
  - [x] comment translation
  - [x] submodule update
  - [x] close inactive issue
