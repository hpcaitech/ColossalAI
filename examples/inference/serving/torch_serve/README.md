# Colossal-Inference with TorchServe

## Overview

This demo is used for testing and demonstrating the usage of Colossal Inference from `colossalai.inference` with deployment with TorchServe. It imports inference modules from colossalai and is based on
https://github.com/hpcaitech/ColossalAI/tree/3e05c07bb8921f2a8f9736b6f6673d4e9f1697d0. For now, single-gpu inference serving is supported.

## Conda Environment for testing
Records to create a conda env to test locally as follows. We might want to use docker or configure env on cloud platform later.

*NOTE*: It requires the installation of jdk and the set of `JAVA_HOME`. We recommend to install open-jdk-17 (Please refer to https://openjdk.org/projects/jdk/17/)

```bash
# use python 3.8 or 3.9
conda create -n infer python=3.9

# use torch 1.13+cuda11.6 for inference
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# conda cuda toolkit (e.g. nvcc, etc)
conda install -c "nvidia/label/cuda-11.6.2" cuda-toolkit

# install colossalai with PyTorch extensions
cd <path_to_ColossalAI_repo>
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-test.txt
CUDA_EXT=1 pip install -e .

# install torchserve
cd <path_to_pytorch_serve_repo>
python ./ts_scripts/install_dependencies.py --cuda=cu116
pip install torchserve torch-model-archiver torch-workflow-archiver
```

## Steps to deploy a model

###  1.download/prepare a model
To use on cloud platform, we will zip the downloaded model.
```bash
# download snapshots
huggingface-cli login
python download_model.py --model_name bigscience/bloom-560m -o <path_to_store_downloaded_model>

# zip the model repo
cd <path_to_store_downloaded_model>/models--bigscience--bloom-560m/snapshots/<specific_revision>
zip -r <path_to_place_zipped_model>//model.zip *
```

> **_NOTE:_**  The torch archiver and server will use `/tmp/` folder. Depending on the limit of disk quota, using torch-model-archiver might cause OSError "Disk quota exceeded". To prevent the OSError, set tmp dir environment variable as follows:
`export TMPDIR=<dir_with_enough_space>/tmp` and `export TEMP=<dir_with_enough_space>/tmp`,
or use relatively small models (as we did) for local testing.

### 2. Archive the model
With torch archiver, we will pack the model file (.zip) as well as handler file (.py) together into a .mar file. And then in serving process these files will be unpacked by TorchServe. Revelant model configs and inference configs can be set in `model-config.yaml`.
```bash
cd ./ColossalAI/examples/inference/serving/torch_serve
# create a folder under the current directory to store the packed model created by torch archiver
mkdir model_store
torch-model-archiver --model-name bloom --version 0.1 --handler Colossal_Inference_Handler.py --config-file model-config.yaml --extra-files <dir_zipped_model>/model.zip --export-path ./model_store/
```

### 3. Launch serving

Modify `load_models` in config.properties to select the model(s) stored in <model_store> directory to be deployed. By default we use `load_models=all` to load and deploy all the models (.mar) we have.

```bash
torchserve --start --ncs --ts-config config.properties
```
We could set inference, management, and metrics addresses and other TorchServe settings in `config.properties`.

TorchServe will create a folder `logs/` under the current directory to store ts, model, and metrics logs.

### 4. Run inference

```bash
# check inference status
curl http://0.0.0.0:8084/ping

curl -X POST http://localhost:8084/predictions/bloom -T sample_text.txt
```

To stop TorchServe, run `torchserve --stop`
