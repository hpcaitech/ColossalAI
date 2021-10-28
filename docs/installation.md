# Setup

## Install with pip

```bash
pip install colossalai
```

## Install from source

```shell
git clone git@github.com:hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

Install and enable CUDA kernel fusion (compulsory installation when using fused optimizer)

```
pip install -v --no-cache-dir --global-option="--cuda_ext" .
```
