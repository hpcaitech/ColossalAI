# Setup

### PyPI

```bash
pip install colossalai
```

### Install From Source (Recommended)

> We **recommend** you to install from source as the Colossal-AI is updating frequently in the early versions. The documentation will be in line with the main branch of the repository. Feel free to raise an issue if you encounter any problem. :)

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

Install and enable CUDA kernel fusion (compulsory installation when using fused optimizer)

```shell
pip install -v --no-cache-dir --global-option="--cuda_ext" .
```
