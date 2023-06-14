# Contributing

Colossal-AI welcomes any constructive contribution from the community and the team is more than willing to work on problems you have encountered to make it a better project.

## Environment Setup

To contribute to Colossal-AI, we would like to first guide you to set up a proper development environment so that you can better implement your code. It is good to install this system from source with the `editable` flag (`-e`, for development mode) so that your change to the source code will be reflected in runtime without repeated installation and uninstallation. Here are the steps to set up the development environment.

1. Uninstall any existing Colossal-AI distribution.

```shell
pip uninstall colossalai
```

2. Clone the repository to local workspace

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
```

3. The *Get Started* section of [official documentation](https://colossalai.org) has provided instructions to build from source. Follow to instruction to build from source, **but replace the last `pip install` statement with the command below by adding the `-e` flag.**

```shell
pip install <options> -e .
```

## Coding Standards

### Unit Tests
We use [PyTest](https://docs.pytest.org/en/latest/) to execute tests. You can install pytest by `pip install pytest`. As some of the tests require initialization of the distributed backend, GPUs are needed to execute these tests.

To set up the environment for unit testing, first change your current directory to the root directory of your local ColossalAI repository, then run
```bash
pip install -r requirements/requirements-test.txt
```
If you encounter an error telling "Could not find a version that satisfies the requirement fbgemm-gpu==0.2.0", please downgrade your python version to 3.8 or 3.9 and try again.

If you only want to run CPU tests, you can run

```bash
pytest -m cpu tests/
```

If you have 8 GPUs on your machine, you can run the full test

```bash
pytest tests/
```

If you do not have 8 GPUs on your machine, do not worry. Unit testing will be automatically conducted when you put up a pull request to the main branch.


### Code Style

We have some static checks when you commit your code change, please make sure you can pass all the tests and make sure the coding style meets our requirements. We use pre-commit hook to make sure the code is aligned with the writing standard. To set up the code style checking, you need to follow the steps below.

```shell
# these commands are executed under the Colossal-AI directory
pip install pre-commit
pre-commit install
```

Code format checking will be automatically executed when you commit your changes.


## Contribution Guide

You need to follow these steps below to make contribution to the main repository via pull request. You can learn about the details of pull request [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

### 1. Fork the Official Repository

Firstly, you need to visit the [Colossal-AI repository](https://github.com/hpcaitech/ColossalAI) and fork into your own account. The `fork` button is at the right top corner of the web page alongside with buttons such as `watch` and `star`.

Now, you can clone your own forked repository into your local environment.

```shell
git clone https://github.com/<YOUR-USERNAME>/ColossalAI.git
```

### 2. Configure Git

You need to set the official repository as your upstream so that you can synchronize with the latest update in the official repository. You can learn about upstream [here](https://www.atlassian.com/git/tutorials/git-forks-and-upstreams).

Then add the original repository as upstream

```shell
cd ColossalAI
git remote add upstream https://github.com/hpcaitech/ColossalAI.git
```

you can use the following command to verify that the remote is set. You should see both `origin` and `upstream` in the output.

```shell
git remote -v
```

### 3. Synchronize with Official Repository

Before you make changes to the codebase, it is always good to fetch the latest updates in the official repository. In order to do so, you can use the commands below.

```shell
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

Otherwise, you can click the `fetch upstream` button on the github webpage of the main branch of your forked repository. Then, use these commands to sync.

```
git checkout main
git fetch main
```

### 4. Choose/Create an Issue for Your Pull Request

Generally, your code change should be only targeted at one problem. Stacking multiple commits for different problems into one pull request will only make the code review such dire suffering and make the system prone to new bugs as the reviewer may not understand the code logic correctly. Thus, you should choose an existing issue or [create your own issue](https://github.com/hpcaitech/ColossalAI/issues) as your pull request target. If you wish to create a new issue, do use appropriate title and description and add related labels.


### 5. Create a New Branch

You should not make changes to the `main` branch of your forked repository as this might make upstream synchronization difficult. You can create a new branch with the appropriate name. General branch name format should start with `hotfix/` and `feature/`. `hotfix` is for bug fix and `feature` is for addition of a new feature.


```shell
git checkout -b <NEW-BRANCH-NAME>
```

### 6. Implementation and Code Commit

Now you can implement your code change in the source code. Remember that you installed the system in development, thus you do not need to uninstall and install to make the code take effect. The code change will be reflected in every new PyThon execution.
You can commit and push the changes to your local repository. The changes should be kept logical, modular and atomic.

```shell
git add -A
git commit -m "<COMMIT-MESSAGE>"
git push -u origin <NEW-BRANCH-NAME>
```

### 7. Open a Pull Request

You can now create a pull request on the GitHub webpage of your repository. The source branch is `<NEW-BRANCH-NAME>` of your repository and the target branch should be `main` of `hpcaitech/ColossalAI`. After creating this pull request, you should be able to see it [here](https://github.com/hpcaitech/ColossalAI/pulls).

Do write clearly the description of your pull request and [link the pull request to your target issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue). This will automatically close the issue when the pull request is approved.

In case of code conflict, you should rebase your branch and resolve the conflicts manually.
