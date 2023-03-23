# Colossal-AI Examples

## Table of Contents

- [Colossal-AI Examples](#colossal-ai-examples)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Folder Structure](#folder-structure)
  - [Integrate Your Example With Testing](#integrate-your-example-with-testing)

## Overview

This folder provides several examples accelerated by Colossal-AI. The `tutorial` folder is for everyone to quickly try out the different features in Colossal-AI. Other folders such as `images` and `language` include a wide range of deep learning tasks and applications.

## Folder Structure

```text
└─ examples
  └─ images
      └─ vit
        └─ test_ci.sh
        └─ train.py
        └─ README.md
      └─ ...
  └─ ...
```
## Invitation to open-source contribution
Referring to the successful attempts of [BLOOM](https://bigscience.huggingface.co/) and [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion), any and all developers and partners with computing powers, datasets, models are welcome to join and build the Colossal-AI community, making efforts towards the era of big AI models!

You may contact us or participate in the following ways:
1. [Leaving a Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) to show your like and support. Thanks!
2. Posting an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose), or submitting a PR on GitHub follow the guideline in [Contributing](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md).
3. Join the Colossal-AI community on
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w),
and [WeChat(微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") to share your ideas.
4. Send your official proposal to email contact@hpcaitech.com

Thanks so much to all of our amazing contributors!

## Integrate Your Example With Testing

Regular checks are important to ensure that all examples run without apparent bugs and stay compatible with the latest API.
Colossal-AI runs workflows to check for examples on a on-pull-request and weekly basis.
When a new example is added or changed, the workflow will run the example to test whether it can run.
Moreover, Colossal-AI will run testing for examples every week.

Therefore, it is essential for the example contributors to know how to integrate your example with the testing workflow. Simply, you can follow the steps below.

1. Create a script called `test_ci.sh` in your example folder
2. Configure your testing parameters such as number steps, batch size in `test_ci.sh`, e.t.c. Keep these parameters small such that each example only takes several minutes.
3. Export your dataset path with the prefix `/data` and make sure you have a copy of the dataset in the `/data/scratch/examples-data` directory on the CI machine. Community contributors can contact us via slack to request for downloading the dataset on the CI machine.
4. Implement the logic such as dependency setup and example execution
