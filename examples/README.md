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
