## Overview
This example shows how to use ColossalAI to run huggingface GPT training in distributed manners.

## GPT
We use the huggingface transformers GPT2 model. The input data is randonly generated.

## Our Modifications
We adapt the OPT training code to ColossalAI by leveraging Gemini and ZeRO DDP.

## Quick Start
You can launch training by using the following bash script

```bash
pip install -r requirements.txt
bash run.sh
```
