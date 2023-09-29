# Colossal-Inference with Ray Serve

This example is used for demonstrating and testing the deployment of Colossal Inference from `colossalai.inference` with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). It imports inference modules from colossalai and is based on https://github.com/hpcaitech/ColossalAI/tree/a22706337a57dd1c98b95739dd09d98bd55947a0.

Single-gpu inference as well as multiple-gpu inference (i.e. tensor parallel) serving are supported.

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

# install other dependencies
pip install triton==2.0.0.dev20221202
pip install transformers
```

## Launch Ray Serve and run the app
### Method #1. CLI command

Under the current directory, we could launch the app by the following command:
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
We could also launch ray serve and run the app inside the script.
Attach the following to the end of the file,
```pyhton
handle: DeploymentHandle = serve.run(app)
print(requests.get("http://localhost:8000/?text={}".format(text)))
```
and then run the script by `python Colossal_Inference_rayserve.py`


### Terminate Ray Serve
Ray serve and the application would terminate automatically as you choose the second method to run any job in the script. If you choose the first method (serve run), you might want to apply `ctrl+c` to shut down the application.

To make sure all the active Ray processes are killed, run
```bash
ray stop
```
