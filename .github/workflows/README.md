# CI/CD

## Table of Contents

- [CI/CD](#cicd)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Workflows](#workflows)
    - [Checks on Pull Requests](#checks-on-pull-requests)
    - [Regular Checks](#regular-checks)
    - [Release](#release)
    - [Manual Dispatch](#manual-dispatch)
      - [Release bdist wheel](#release-bdist-wheel)
      - [Dispatch Example Test](#dispatch-example-test)
      - [Compatibility Test](#compatibility-test)
    - [User Friendliness](#user-friendliness)
  - [Progress Log](#progress-log)

## Overview

Automation makes our development more efficient as the machine automatically run the pre-defined tasks for the contributors.
This saves a lot of manual work and allow the developer to fully focus on the features and bug fixes.
In Colossal-AI, we use [GitHub Actions](https://github.com/features/actions) to automate a wide range of workflows to ensure the robustness of the software.
In the section below, we will dive into the details of different workflows available.

## Workflows

### Checks on Pull Requests

| Workflow Name               | File name                      | Description                                                                                                                                       |
| --------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Build`                     | `build.yml`                    | This workflow is triggered when the label `Run build and Test` is assigned to a PR. It will run all the unit tests in the repository with 4 GPUs. |
| `Pre-commit`                | `pre_commit.yml`               | This workflow runs pre-commit checks for code style consistency.                                                                                  |
| `Report pre-commit failure` | `report_precommit_failure.yml` | This PR will put up a comment in the PR to explain the precommit failure and remedy. This is executed when `Pre-commit` is done                   |
| `Report test coverage`      | `report_test_coverage.yml`     | This PR will put up a comment to report the test coverage results. This is executed when `Build` is completed.                                    |
| `Test example`              | `auto_example_check.yml`       | The example will be automatically tested if its files are changed in the PR                                                                       |

### Regular Checks

| Workflow Name           | File name                | Description                                                                                                            |
| ----------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `Test example`          | `auto_example_check.yml` | This workflow will test all examples every Sunday                                                                      |
| `Build on 8 GPUs`       | `build_gpu_8.yml`        | This workflow will run the unit tests everyday with 8 GPUs.                                                            |
| `Synchronize submodule` | `submodule.yml`          | This workflow will check if any git submodule is updated. If so, it will create a PR to update the submodule pointers. |
| `Close inactive issues` | `close_inactive.yml`     | This workflow will close issues which are stale for 14 days.                                                           |

### Release

| Workflow Name               | File name                       | Description                                                                                                       |
| --------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `Draft GitHub Release Post` | `draft_github_release_post.yml` | Compose a GitHub release post draft based on the commit history. Triggered when `version.txt` is updated.         |
| `Release to PyPI`           | `release_pypi.yml`              | Build and release the wheel to PyPI. Triggered when `version.txt` is updated.                                     |
| `Release Nightly to PyPI`   | `release_nightly.yml`           | Build and release the nightly wheel to PyPI as `colossalai-nightly`. Automatically executed every Sunday.         |
| `Release Docker`            | `release_docker.yml`            | Build and release the Docker image to DockerHub. Triggered when `version.txt` is updated.                         |
| `Release bdist wheel`       | `release_bdist.yml`             | Build binary wheels with pre-built PyTorch extensions. Manually dispatched. See more details in the next section. |

### Manual Dispatch

| Workflow Name           | File name                    | Description                                            |
| ----------------------- | ---------------------------- | ------------------------------------------------------ |
| `Release bdist wheel`   | `release_bdist.yml`          | Build binary wheels with pre-built PyTorch extensions. |
| `Dispatch Example Test` | `dispatch_example_check.yml` | Manually test a specified example.                     |
| `Compatiblity Test`     | `compatiblity_test.yml`      | Test PyTorch and Python Compatibility.                 |

Refer to this [documentation](https://docs.github.com/en/actions/managing-workflow-runs/manually-running-a-workflow) on how to manually trigger a workflow.
I will provide the details of each workflow below.

#### Release bdist wheel

Parameters:
- `torch version`:torch version to test against, multiple versions are supported but must be separated by comma. The default is value is all, which will test all available torch versions listed in this [repository](https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_build/torch_wheels) which is regularly updated.
- `cuda version`: cuda versions to test against, multiple versions are supported but must be separated by comma. The CUDA versions must be present in our [DockerHub repository](https://hub.docker.com/r/hpcaitech/cuda-conda).
- `ref`: input the branch or tag name to build the wheel for this ref.

#### Dispatch Example Test

parameters:
- `example_directory`: the example directory to test. Multiple directories are supported and must be separated by comma. For example, language/gpt, images/vit. Simply input language or simply gpt does not work.


#### Compatibility Test

Parameters:
- `torch version`:torch version to test against, multiple versions are supported but must be separated by comma. The default is value is all, which will test all available torch versions listed in this [repository](https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_build/torch_wheels).
- `cuda version`: cuda versions to test against, multiple versions are supported but must be separated by comma. The CUDA versions must be present in our [DockerHub repository](https://hub.docker.com/r/hpcaitech/cuda-conda).

> It only test the compatiblity of the main branch


### User Friendliness

| Workflow Name     | File name               | Description                                                                                                                            |
| ----------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `issue-translate` | `translate_comment.yml` | This workflow is triggered when a new issue comment is created. The comment will be translated into English if not written in English. |

## Progress Log

- [x] unit testing
  - [x] test on PR
  - [x] report test coverage
  - [x] regular test
- [x] release
  - [x] official release
  - [x] nightly build
  - [x] binary build
  - [x] docker build
  - [x] draft release post
- [x] pre-commit
  - [x] check on PR
  - [x] report failure
- [x] example check
  - [x] check on PR
  - [x] regular check
  - [x] manual dispatch
- [ ] compatiblity check
  - [x] manual dispatch
  - [ ] auto test when release
- [x] helpers
  - [x] comment translation
  - [x] submodule update
  - [x] close inactive issue
