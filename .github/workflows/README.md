# CI/CD

## Table of Contents

- [CI/CD](#cicd)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Workflows](#workflows)
    - [Code Style Check](#code-style-check)
    - [Unit Test](#unit-test)
    - [Example Test](#example-test)
      - [Dispatch Example Test](#dispatch-example-test)
    - [Compatibility Test](#compatibility-test)
      - [Compatibility Test](#compatibility-test-1)
    - [Release](#release)
      - [Release bdist wheel](#release-bdist-wheel)
    - [User Friendliness](#user-friendliness)
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

### Code Style Check

| Workflow Name               | File name                      | Description                                                                                                |
| --------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `Pre-commit`                | `pre_commit.yml`               | This workflow runs pre-commit checks for code style consistency for PRs.                                   |
| `Report pre-commit failure` | `report_precommit_failure.yml` | This PR will put up a comment in the PR to explain the precommit failure and remedy if `Pre-commit` fails. |

### Unit Test

| Workflow Name          | File name                  | Description                                                                                                                                       |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Build`                | `build.yml`                | This workflow is triggered when the label `Run build and Test` is assigned to a PR. It will run all the unit tests in the repository with 4 GPUs. |
| `Build on 8 GPUs`      | `build_gpu_8.yml`          | This workflow will run the unit tests everyday with 8 GPUs.                                                                                       |
| `Report test coverage` | `report_test_coverage.yml` | This PR will put up a comment to report the test coverage results when `Build` is done.                                                           |

### Example Test

| Workflow Name              | File name                       | Description                                                                 |
| -------------------------- | ------------------------------- | --------------------------------------------------------------------------- |
| `Test example on PR`       | `example_check_on_pr.yml`       | The example will be automatically tested if its files are changed in the PR |
| `Test example on Schedule` | `example_check_on_schedule.yml` | This workflow will test all examples every Sunday                           |
| `Example Test on Dispatch` | `example_check_on_dispatch.yml` | Manually test a specified example.                                          |

#### Dispatch Example Test

parameters:
- `example_directory`: the example directory to test. Multiple directories are supported and must be separated by comma. For example, language/gpt, images/vit. Simply input language or simply gpt does not work.

### Compatibility Test

| Workflow Name                | File name                        | Description                                                                                                                   |
| ---------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `Compatibility Test`         | `auto_compatibility_test.yml`    | This workflow will check the compatiblity of Colossal-AI against PyTorch and CUDA specified in `.compatibility` every Sunday. |
| `Auto Compatibility Test`    | `auto_compatibility_test.yml`    | Check Colossal-AI's compatiblity when `version.txt` is changed in a PR.                                                       |
| `Dispatch Compatiblity Test` | `dispatch_compatiblity_test.yml` | Test PyTorch and Python Compatibility.                                                                                        |


#### Compatibility Test

Parameters:
- `torch version`:torch version to test against, multiple versions are supported but must be separated by comma. The default is value is all, which will test all available torch versions listed in this [repository](https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_build/torch_wheels).
- `cuda version`: cuda versions to test against, multiple versions are supported but must be separated by comma. The CUDA versions must be present in our [DockerHub repository](https://hub.docker.com/r/hpcaitech/cuda-conda).

> It only test the compatiblity of the main branch


### Release

| Workflow Name               | File name                       | Description                                                                                                                                                 |
| --------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Draft GitHub Release Post` | `draft_github_release_post.yml` | Compose a GitHub release post draft based on the commit history.  Triggered when the change of `version.txt` is merged.                                     |
| `Release to PyPI`           | `release_pypi.yml`              | Build and release the wheel to PyPI.  Triggered when the change of `version.txt` is merged.                                                                 |
| `Release Nightly to PyPI`   | `release_nightly.yml`           | Build and release the nightly wheel to PyPI as `colossalai-nightly`. Automatically executed every Sunday.                                                   |
| `Release Docker`            | `release_docker.yml`            | Build and release the Docker image to DockerHub. Triggered when the change of `version.txt` is merged.                                                      |
| `Release bdist wheel`       | `release_bdist.yml`             | Build binary wheels with pre-built PyTorch extensions. Manually dispatched. See more details in the next section.                                           |
| `Auto Release bdist wheel`  | `auto_release_bdist.yml`        | Build binary wheels with pre-built PyTorch extensions.Triggered when the change of `version.txt` is merged. Build specificatons are stored in `.bdist.json` |
| `Release bdist wheel`       | `release_bdist.yml`             | Build binary wheels with pre-built PyTorch extensions.                                                                                                      |


#### Release bdist wheel

Parameters:
- `torch version`:torch version to test against, multiple versions are supported but must be separated by comma. The default is value is all, which will test all available torch versions listed in this [repository](https://github.com/hpcaitech/public_assets/tree/main/colossalai/torch_build/torch_wheels) which is regularly updated.
- `cuda version`: cuda versions to test against, multiple versions are supported but must be separated by comma. The CUDA versions must be present in our [DockerHub repository](https://hub.docker.com/r/hpcaitech/cuda-conda).
- `ref`: input the branch or tag name to build the wheel for this ref.

### User Friendliness

| Workflow Name           | File name               | Description                                                                                                                            |
| ----------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `issue-translate`       | `translate_comment.yml` | This workflow is triggered when a new issue comment is created. The comment will be translated into English if not written in English. |
| `Synchronize submodule` | `submodule.yml`         | This workflow will check if any git submodule is updated. If so, it will create a PR to update the submodule pointers.                 |
| `Close inactive issues` | `close_inactive.yml`    | This workflow will close issues which are stale for 14 days.                                                                           |


## Configuration

This section lists the files used to configure the workflow.

1. `.compatibility`

This `.compatibility` file is to tell GitHub Actions which PyTorch and CUDA versions to test against. Each line in the file is in the format `${torch-version}-${cuda-version}`, which is a tag for Docker image. Thus, this tag must be present in the [docker registry](https://hub.docker.com/r/pytorch/conda-cuda) so as to perform the test.

2. `.bdist.json`

This file controls what pytorch/cuda compatible pre-built releases will be built and published. You can add a new entry according to the json schema below if there is a new wheel that needs to be built with AOT compilation of PyTorch extensions.

```json
{
  "build": [
    {
      "torch_version": "",
      "cuda_image": ""
    },
  ]
}
```

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
- [x] compatiblity check
  - [x] manual dispatch
  - [x] auto test when release
- [x] helpers
  - [x] comment translation
  - [x] submodule update
  - [x] close inactive issue
