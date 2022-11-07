# ColoDiffusion
*ColoDiffusion is a Faster Train implementation of the model [stable-diffusion](https://github.com/CompVis/stable-diffusion) from [Stability AI](https://stability.ai/)*

We take advantage of Colosssal-AI to exploit multiple optimization strategies
, e.g. data parallelism, tensor parallelism, mixed precision & ZeRO, to scale the training to multiple GPUs.



![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1) is a latent text-to-image diffusion
model.
Thanks to a generous compute donation from [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), we were able to train a Latent Diffusion Model on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database.
Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487),
this model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts.
With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.
See [this section](#stable-diffusion-v1) below and the [model card](https://huggingface.co/CompVis/stable-diffusion).


## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
```

### Install ColossalAI

```
git clone https://github.com/hpcaitech/ColossalAI.git
git checkout v0.1.10
pip install .
```

## Training

we provide the script `train.sh` to run the training task , and three Stategy in `configs`:`train_colossalai.yaml`, `train_ddp.yaml`, `train_deepspeed.yaml`

for example, you can run the training from colossalai by
```
python main.py --logdir /tmp -t --postfix test -b config/train_colossalai.yaml
```

you can change the trainging config in the yaml file

- accelerator: acceleratortype, default 'gpu'
- devices: device number used for training, default 4
- max_epochs: max training epochs
- precision: usefp16 for training or not, default 16, you must use fp16 if you want to apply colossalai


## Comments

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
and [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories).

- the implementation of [flash attention](https://github.com/HazyResearch/flash-attention) is from [HazyResearch](https://github.com/HazyResearch)

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models},
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@article{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2205.14135},
  year={2022}
}
```
