# Colossal-Inference


## Table of Contents

-   📚 Introduction

- 🔨 Installation
- 🚀 Quick Start
- 💡Usage

## 📚 Introduction

This example lets you to set up and quickly try out our Colossal-Inference.

## 🔨 Installation

### Install From Source

Prerequistes:

-   Python == 3.9
-   PyTorch >= 2.1.0
-   CUDA == 11.8
-   Linux OS

We strongly recommend you use [Anaconda](https://www.anaconda.com/) to create a new environment (Python >= 3.9) to run our examples:

```shell
# Create a new conda environment
conda create -n inference python=3.9 -y
conda activate inference
```

Install the latest PyTorch (with CUDA == 11.8) using conda:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install Colossal-AI from source:

```shell
# Clone Colossal-AI repository to your workspace
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# Install Colossal-AI from source
CUDA_EXT=1 pip install .
```

Install inference dependencies:

```shell
# Install inference dependencies
pip install -r requirements/requirements-infer.txt
```

**(Optional)** If you want to use [SmoothQuant](https://github.com/mit-han-lab/smoothquant) quantization, you need to install `torch-int` following this [instruction](https://github.com/Guangxuan-Xiao/torch-int#:~:text=cmake%20%3E%3D%203.12-,Installation,-git%20clone%20%2D%2Drecurse).

### Use Colossal-Inference in Docker

#### Pull from DockerHub

You can directly pull the docker image from our [DockerHub page](https://hub.docker.com/r/hpcaitech/colossalai). The image is automatically uploaded upon release.

```shell
docker pull hpcaitech/colossalai-inference:v2
```

#### Build On Your Own

Run the following command to build a docker image from Dockerfile provided.

>   Building Colossal-AI from scratch requires GPU support, you need to use Nvidia Docker Runtime as the default when doing `docker build`. More details can be found [here](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime). We recommend you install Colossal-AI from our [project page](https://www.colossalai.org/) directly.

```shell
cd ColossalAI/inference/dokcer
docker build
```

Run the following command to start the docker container in interactive mode.

```shell
docker run -it --gpus all --name ANY_NAME -v $PWD:/workspace -w /workspace hpcaitech/colossalai-inference:v2 /bin/bash
```

\[Todo\]: Waiting for Docker file (Li Cuiqing)

## 🚀 Quick Start

You can try the example following the instructions below:

```shell
cd ColossalAI/examples/inference
python run_llama_inference.py -p [xxxxx] -b 4 --max_input_len 128 --max_output_len 64 --dtype fp16
```

\[Todo\]: Waiting for new examples (quant: Xu Kai; Inference output: Zhao Yuanheng)

## 💡 Usage

A general way to use Colossal-Inference will be:

```python
# Import required modules
import ...

# Prepare your model
model = ...

# Declare configurations
tp_size = ...
pp_size = ...
...

# Create an inference engine
engine = InferenceEngine(model, [tp_size, pp_size, ...])

# Tokenize the input
inputs = ...

# Perform inferencing based on the inputs
outputs = engine.generate(inputs)
```
