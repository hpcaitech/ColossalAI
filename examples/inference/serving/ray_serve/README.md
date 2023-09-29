# Colossal-Inference with Ray Serve

This example is used for demonstrating and testing the deployment of Colossal Inference from `colossalai.inference` with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). It imports inference modules from colossalai and is based on https://github.com/hpcaitech/ColossalAI/tree/a22706337a57dd1c98b95739dd09d98bd55947a0.

Single-gpu inference and multiple-gpu inference (i.e. tensor parallel) serving are supported.

## Env Installation

Conda
```bash
# create a new conda env with python 3.8
conda create -n ray_test python=3.8.18

# use torch1.13+cuda11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install ray from wheels
pip install -U "ray[default,serve]"

# install cuda toolkit (e.g. nvcc, etc)
conda install -c "nvidia/label/cuda-11.6.2" cuda-toolkit

# install cuDNN, cuTENSOR, and NCCL
conda install -c conda-forge cupy cudnn cutensor nccl cuda-version=11.6

# install colossalai with PyTorch extensions
cd <path_to_ColossalAI_repo>
CUDA_EXT=1 pip install -e .

pip install transformers
```

## Launch Ray Serve and run the app
### Method #1. CLI command

```bash
    RAY_DEDUP_LOGS=0 serve run Colossal_Inference_rayserve:app
```

By default, Ray deduplicates logs across cluster. Here we set `RAY_DEDUP_LOGS=0`` to disable log deduplication, enabling each actor to log information in CLI.

`serve run` runs an application from the specified import path. The formats should be `<filename>:<app_name>`.

Then we could send requests by running python script in another window:
```bash
python send_request.py
```

### Method #2. Run inside script

Attach the following to the end of the file, and run the script via `python Colossal_Inference_rayserve.py`
```pyhton
handle: DeploymentHandle = serve.run(app)
print(requests.get("http://localhost:8000/?text={}".format(text)))
```



Use
```bash
ray stop
```

to kill any active Ray processes
