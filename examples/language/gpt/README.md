## Overview
This example shows how to use ColossalAI to run huggingface GPT training in distributed manners.

## GPT
We use the huggingface transformers GPT2 model. The input data is randonly generated.

## Our Modifications
The `train_gpt_demo.py` provides three distributed plans, i.e. ColossalAI, PyTorch DDP and ZeRO.
The ColossalAI leverages Tensor Parallel and Gemini.

## Quick Start
You can launch training by using the following bash script

```bash
pip install -r requirements.txt
bash run.sh
```
