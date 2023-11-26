# Command Line Tool

Author: Shenggui Li

**Prerequisite:**
- [Distributed Training](../concepts/distributed_training.md)
- [Colossal-AI Overview](../concepts/colossalai_overview.md)

## Introduction

Colossal-AI provides command-line utilities for the user.
The current command line tools support the following features.

- verify Colossal-AI build
- launch distributed jobs
- tensor parallel micro-benchmarking

## Check Installation

To verify whether your Colossal-AI is built correctly, you can use the command `colossalai check -i`.
This command will inform you information regarding the version compatibility and cuda extension.

<figure style={{textAlign: "center"}}>
<img src="https://s2.loli.net/2022/05/04/KJmcVknyPHpBofa.png"/>
<figcaption>Check Installation Demo</figcaption>
</figure>

## Launcher

To launch distributed jobs on single or multiple nodes, the command `colossalai run` can be used for process launching.
You may refer to [Launch Colossal-AI](./launch_colossalai.md) for more details.

<!-- doc-test-command: echo  -->
