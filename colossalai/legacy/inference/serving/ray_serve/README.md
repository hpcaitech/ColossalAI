# Colossal-Inference with Ray Serve

This example is used for demonstrating and testing the deployment of Colossal Inference from `colossalai.inference` with [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). It imports inference modules from colossalai and is based on https://github.com/hpcaitech/ColossalAI/tree/a22706337a57dd1c98b95739dd09d98bd55947a0.

Single-gpu inference as well as multiple-gpu inference (i.e. tensor parallel) serving are supported.

## Installation

### Conda Environment
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
BUILD_EXT=1 pip install -e .

# install other dependencies
pip install triton==2.0.0.dev20221202
pip install transformers
```

## Launch Ray Serve and run the app
### Method #1. CLI command

Under the current directory, we could launch the app by the following command:
```bash
RAY_DEDUP_LOGS=0 serve run Colossal_Inference_rayserve:app path="PATH_TO_YOUR_MODEL_DIR"
```

By default, Ray deduplicates logs across cluster. Here we set `RAY_DEDUP_LOGS=0` to disable log deduplication, enabling each actor to log information in CLI. `serve run` runs an application from the specified import path. The formats should be `<filename>:<app_name>`.

Then we could send requests by running python script in another window:
```bash
python send_request.py
```

### Method #2. Run inside script

We could also launch ray serve and run the app inside a single script by making some modifications:
To avoid ray handler from raising error in serializing pydantic objects, we'll replace the config class from `class GenConfigArgs(BaseModel)` to
```python
from dataclasses import dataclass
@dataclass
class GenConfigArgs:
    # attributes remain unchanged
```
Comment out the app builder
```python
# def app(args: GenConfigArgs) -> Application:
#     ...
#     return Driver.options(name="Colossal-Inference-Driver").bind(config=args)
```
And attach the following lines to the end of the file,
```python
from ray.serve.handle import DeploymentHandle, DeploymentResponse

app = Driver.bind(config=GenConfigArgs(path="<Path_to_model_dir>"))
handle: DeploymentHandle = serve.run(app).options(use_new_handle_api=True)
response: DeploymentResponse = handle.batch_generate.remote(requests="Introduce some landmarks in Beijing")
print(response.result())
```
Then we could run the script
```python
python Colossal_Inference_rayserve.py
```

### Terminate Ray Serve
Ray serve and the application would terminate automatically as you choose the second method to run any job in the script. If you choose the first method (serve run), you might want to apply `ctrl+c` to shut down the application, or use `serve shutdown` to shut down serve and deletes all applications on the ray cluster.

To make sure all the active Ray processes are killed, run
```bash
ray stop
```
