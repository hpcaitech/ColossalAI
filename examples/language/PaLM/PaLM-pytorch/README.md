<img src="./palm.gif" width="450px"></img>

## PaLM - Pytorch

Implementation of the specific Transformer architecture from <a href="https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html">PaLM - Scaling Language Modeling with Pathways</a>, in less than <a href="https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py">200 lines of code</a>.

This model is pretty much SOTA on everything language. <a href="https://www.youtube.com/watch?v=RJwPN4qNi_Y">Yannic Kilcher explanation</a>

It obviously will not scale, but it is just for educational purposes. To elucidate the public how simple it all really is.

<a href="https://github.com/lucidrains/palm-jax">Jax version</a>

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

## Test on Enwik8

```bash
$ python train.py
```

## Citations

```bibtex
@inproceedings{Chowdhery2022PaLMSL,
    title   = {PaLM: Scaling Language Modeling with Pathways},
    author  = {Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Oliveira Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark Diaz and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
    year    = {2022}
}
```

```bibtex
@article{Tillet2019TritonAI,
    title   = {Triton: an intermediate language and compiler for tiled neural network computations},
    author  = {Philippe Tillet and H. T. Kung and David D. Cox},
    journal = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year    = {2019}
}
```
