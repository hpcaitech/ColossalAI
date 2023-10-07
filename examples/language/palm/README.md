<img src="./palm.gif" width="450px"></img>

## PaLM - Pytorch

Implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a>, in less than 200 lines of code.

This model is pretty much SOTA on everything language.

It obviously will not scale, but it is just for educational purposes. To elucidate the public how simple it all really is.

## Install
```bash
$ pip install PaLM-pytorch
```

## Usage

```python
import torch
from palm_pytorch import PaLM

palm = PaLM(
    num_tokens = 20000,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64,
)

tokens = torch.randint(0, 20000, (1, 2048))
logits = palm(tokens) # (1, 2048, 20000)
```

The PaLM 540B in the paper would be

```python
palm = PaLM(
    num_tokens = 256000,
    dim = 18432,
    depth = 118,
    heads = 48,
    dim_head = 256
)
```

## New API
We have modified our previous implementation of PaLM with our new Booster API, which offers a more flexible and efficient way to train your model. The new API is more user-friendly and easy to use. You can find the new API in train.py. We also offer a shell script test_ci.sh for you to go through all our plugins for the booster. For more information about the booster API you can refer to https://colossalai.org/docs/basics/booster_api/.

## Test on Enwik8

```bash
$ python train.py
```

## Todo

- [ ] offer a Triton optimized version of PaLM, bringing in https://github.com/lucidrains/triton-transformer

## Citations

```bibtex
@article{chowdhery2022PaLM,
  title   = {PaLM: Scaling Language Modeling with Pathways},
  author  = {Chowdhery, Aakanksha et al},
  year    = {2022}
}
```
