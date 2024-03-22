# Grok-1 Inference

## Install

```bash
# Make sure you install colossalai from the latest source code
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
pip install .
cd examples/language/grok-1
pip install -r requirements.txt
```

## Tokenizer preparation

You should download the tokenizer from the official grok-1 repository.

```bash
wget https://github.com/xai-org/grok-1/raw/main/tokenizer.model
```

## Inference

You need 8x A100 80GB or equivalent GPUs to run the inference.

We provide two scripts for inference. `run_inference_fast.sh` uses tensor parallelism provided by ColossalAI, and it is faster. `run_inference_slow.sh` uses auto device provided by transformers, and it is slower.

Command format:

```bash
./run_inference_fast.sh <model_name_or_path> <tokenizer_path>
./run_inference_slow.sh <model_name_or_path> <tokenizer_path>
```

`model_name_or_path` can be a local path or a model name from Hugging Face model hub. We provided weights on model hub, named `hpcaitech/grok-1`.

Command example:

```bash
./run_inference_fast.sh hpcaitech/grok-1 tokenizer.model
```

It will take 5-10 minutes to load checkpoints. Don't worry, it's not stuck.
