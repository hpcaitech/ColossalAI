# Text-to-Image: Stable Diffusion with Colossal-AI

Acceleration of AIGC (AI-Generated Content) models such as [Text-to-image model](https://huggingface.co/CompVis/stable-diffusion-v1-4).

More details can be found in our [blog of Stable Diffusion v1](https://www.hpc-ai.tech/blog/diffusion-pretraining-and-hardware-fine-tuning-can-be-almost-7x-cheaper) and [blog of Stable Diffusion v2](https://www.hpc-ai.tech/blog/colossal-ai-0-2-0).


## Installation

### Option #1: Install from source
#### Step 1: Requirements

To begin with, make sure your operating system has the cuda version suitable for this exciting training session, which is cuda11.6/11.8. For your convience, we have set up the rest of packages here. You can create and activate a suitable [conda](https://conda.io/) environment named `ldm` :

```
conda env create -f environment.yaml
conda activate ldm
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running:

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install transformers diffusers invisible-watermark
```

#### Step 2: Install [Colossal-AI](https://colossalai.org/download/) From Our Official Website

You can install the latest version (0.2.7) from our official website or from source. Notice that the suitable version for this training is colossalai(0.2.5), which stands for torch(1.12.1).

##### Download suggested version for this training

```
pip install colossalai==0.2.5
```

##### Download the latest version from pip for latest torch version

```
pip install colossalai
```

##### From source:

```
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI

# install colossalai
CUDA_EXT=1 pip install .
```

#### Step 3: Accelerate with flash attention by xformers (Optional)

Notice that xformers will accelerate the training process at the cost of extra disk space. The suitable version of xformers for this training process is 0.0.12, which can be downloaded directly via pip. For more release versions, feel free to check its official website: [XFormers](https://pypi.org/project/xformers/)

```
pip install xformers==0.0.12
```


### stable-diffusion-model (Recommended)

For example: You can follow this [link] (https://huggingface.co/CompVis/stable-diffusion-v1-4) to download your model. In our training example, we choose Stable-Diffusion-v1-4 as a demo example to who how to train our model. 

### stable-diffusion-v1-4

```
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
```


## Dataset

The dataSet is from [Dataset-HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:text-to-image&sort=downloads). In our example, we choose lambdalabs/pokemon-blip-captions as a demo example. You can also create your own dataset, but make sure your data set matches with formats in this [website] (https://huggingface.co/docs/diffusers/training/create_dataset). 

## Training

We provide the script `01_trainer_no_colossalai.sh` to run the training task without colossalai. Meanwhile, we also provided script called `02_run_trainer_with_colossalai.sh` to train text-to-image model using colossalai. Also, if you want to LoRA to fine-tune your model, we also provided a bash script called `03_run_trainer_with_colossalai_lora.sh` to fine-tune your model. If you are not familar with, you can check this [website](https://huggingface.co/docs/diffusers/training/lora). There is a simple example below to demonstarte how to launch training. 


```
# your model name, the python script will automaticall download corresponding model from model hub.
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# You can provide official dataset to train your model. You also can provide your own dataset. 
export dataset_name="lambdalabs/pokemon-blip-captions"

torchrun --nproc_per_node 4 stable_diffusion_colossalai_trainer.py \
    --mixed_precision="fp16" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$dataset_name \
    --use_ema \
    --resolution=512 --center_crop --random_flip \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=800 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="sd-pokemon-model" \
    --plugin="gemini" \
    --placement="cuda"
```

### Training config

You can change the training config in the yaml file

- nproc_per_node: device number used for training, default = 4
- precision: the precision type used in training, default = 16 (fp16), you must use fp16 if you want to apply colossalai
- plugin：we support the following training stategies: 'torch_ddp', 'torch_ddp_fp16', 'gemini', 'low_level_zero', and we choose gemini as our demonstration in our example. 
- placement_policy: the training strategy supported by Colossal AI, default = 'cuda', which refers to loading all the parameters into cuda memory. On the other hand, 'cpu' refers to 'cpu offload' strategy while 'auto' enables 'Gemini', both featured by Colossal AI.
- more information about the configuration of ColossalAIStrategy can be found [here](https://pytorch-lightning.readthedocs.io/en/latest/advanced/model_parallel.html#colossal-ai)
- Also, for more arguments info, please check parse_arguments.py file in the current directory.


## Invitation to open-source contribution
Referring to the successful attempts of [BLOOM](https://bigscience.huggingface.co/) and [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion), any and all developers and partners with computing powers, datasets, models are welcome to join and build the Colossal-AI community, making efforts towards the era of big AI models!

You may contact us or participate in the following ways:
1. [Leaving a Star ⭐](https://github.com/hpcaitech/ColossalAI/stargazers) to show your like and support. Thanks!
2. Posting an [issue](https://github.com/hpcaitech/ColossalAI/issues/new/choose), or submitting a PR on GitHub follow the guideline in [Contributing](https://github.com/hpcaitech/ColossalAI/blob/main/CONTRIBUTING.md).
3. Join the Colossal-AI community on
[Slack](https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-z7b26eeb-CBp7jouvu~r0~lcFzX832w),
and [WeChat(微信)](https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png "qrcode") to share your ideas.
4. Send your official proposal to email contact@hpcaitech.com

Thanks so much to all of our amazing contributors!

## Comments

- Our codebase for the diffusion models builds heavily on [OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)
, [lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Lightning](https://github.com/Lightning-AI/lightning) and [Hugging Face](https://huggingface.co/CompVis/stable-diffusion).
Thanks for open-sourcing!

- The implementation of the transformer encoder is from [x-transformers](https://github.com/lucidrains/x-transformers) by [lucidrains](https://github.com/lucidrains?tab=repositories).

- The implementation of [flash attention](https://github.com/HazyResearch/flash-attention) is from [HazyResearch](https://github.com/HazyResearch).

## BibTeX

```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
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
