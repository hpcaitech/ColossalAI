# Grok-1 Inference

An easy-to-use Python + PyTorch + HuggingFace version of 314B Grok-1.
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/examples/language/grok-1)
[[blog]](https://hpc-ai.com/blog/grok-1-of-pytorch-huggingface-version-is-now-available)
[[HuggingFace Grok-1 PyTorch model weights]](https://huggingface.co/hpcai-tech/grok-1)

## Installation

```bash
# Make sure you install colossalai from the latest source code
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
pip install .
cd examples/language/grok-1
pip install -r requirements.txt
```

## Inference

You need 8x A100 80GB or equivalent GPUs to run the inference.

We provide two scripts for inference. `run_inference_fast.sh` uses tensor parallelism provided by ColossalAI, which is faster for generation, while `run_inference_slow.sh` uses auto device provided by transformers, which is relatively slower.

Command example:

```bash
./run_inference_fast.sh <MODEL_NAME_OR_PATH>
./run_inference_slow.sh <MODEL_NAME_OR_PATH>
```

`MODEL_NAME_OR_PATH` can be a model name from Hugging Face model hub or a local path to PyTorch-version model checkpoints. We provided weights on model hub, named `hpcaitech/grok-1`. And you could also download the weights in advance using `git`:
```bash
git lfs install
git clone https://huggingface.co/hpcai-tech/grok-1
```

It will take, depending on your Internet speed, several hours to tens of hours to download checkpoints (about 600G!), and 5-10 minutes to load checkpoints when it's ready to launch the inference. Don't worry, it's not stuck.


## Performance

For request of batch size set to 1 and maximum length set to 100:

| Method                  | Initialization-Duration(sec) | Average-Generation-Latency(sec) |
|-------------------------|------------------------------|---------------------------------|
| ColossalAI              | 431.45                       | 14.92                           |
| HuggingFace Auto-Device | 426.96                       | 48.38                           |
| JAX                     | 147.61                       | 56.25                           |

Tested on 8x80G NVIDIA H800.
