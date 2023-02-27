# Setup
> Colossal-AI currently only supports the Linux operating system and has not been tested on other OS such as Windows and macOS.

## Download From PyPI

You can install Colossal-AI with

```shell
pip install colossalai
```

If you want to build PyTorch extensions during installation, you can use the command below. Otherwise, the PyTorch extensions will be built during runtime.

```shell
CUDA_EXT=1 pip install colossalai
```


## Download From Source

> The version of Colossal-AI will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem. :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

If you don't want to install and enable CUDA kernel fusion (compulsory installation when using fused optimizer):

```shell
CUDA_EXT=1 pip install .
```
