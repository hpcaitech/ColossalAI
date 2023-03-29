# [DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth) by [colossalai](https://github.com/hpcaitech/ColossalAI.git)

[DreamBooth](https://arxiv.org/abs/2208.12242) is a method to personalize text2image models like stable diffusion given just a few(3~5) images of a subject.
The `train_dreambooth_colossalai.py` script shows how to implement the training procedure and adapt it for stable diffusion.

By accommodating model data in CPU and GPU and moving the data to the computing device when necessary, [Gemini](https://www.colossalai.org/docs/advanced_tutorials/meet_gemini), the Heterogeneous Memory Manager of [Colossal-AI](https://github.com/hpcaitech/ColossalAI) can breakthrough the GPU memory wall by using GPU and CPU memory (composed of CPU DRAM or nvme SSD memory) together at the same time. Moreover, the model scale can be further improved by combining heterogeneous training with the other parallel approaches, such as data parallel, tensor parallel and pipeline parallel.

## Installation

To begin with, make sure your operating system has the cuda version suitable for this exciting training session, which is cuda11.6-11.8. Notice that you may want to make sure the module versions suitable for the whole environment. Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install -r requirements.txt
```

### Install [colossalai](https://github.com/hpcaitech/ColossalAI.git)

```bash
pip install colossalai
```

**From source**

```bash
git clone https://github.com/hpcaitech/ColossalAI.git
python setup.py install
```

## Dataset for Teyvat BLIP captions
Dataset used to train [Teyvat characters text to image model](https://github.com/hpcaitech/ColossalAI/tree/main/examples/images/diffusion).

BLIP generated captions for characters images from [genshin-impact fandom wiki](https://genshin-impact.fandom.com/wiki/Character#Playable_Characters)and [biligame wiki for genshin impact](https://wiki.biligame.com/ys/%E8%A7%92%E8%89%B2).

For each row the dataset contains `image` and `text` keys. `image` is a varying size PIL png, and `text` is the accompanying text caption. Only a train split is provided.

The `text` include the tag `Teyvat`, `Name`,`Element`, `Weapon`, `Region`, `Model type`, and `Description`, the `Description` is captioned with the [pre-trained BLIP model](https://github.com/salesforce/BLIP).

## Training

We provide the script `colossalai.sh` to run the training task with colossalai. Meanwhile, we also provided traditional training process of dreambooth, `dreambooth.sh`, for possible comparation. For instance, the script of training process for [stable-diffusion-v1-4] model can be modified into:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

torchrun --nproc_per_node 2 train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --placement="cuda"
```
- `MODEL_NAME` refers to the model you are training.
- `INSTANCE_DIR` refers to personalized path to instance images, you might need to insert information here.
- `OUTPUT_DIR` refers to local path to save the trained model, you might need to find a path with enough space.
- `resolution` refers to the corresponding resolution number of your target model. Note: Change the `resolution` to 768 if you are using the [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) 768x768 model.
- `placement`  refers to the training strategy supported by Colossal AI, defult = 'cuda', which refers to loading all the parameters into cuda memory. On the other hand, 'cpu' refers to 'cpu offload' strategy while 'auto' enables 'Gemini', both featured by Colossal AI.

### Training with prior-preservation loss

Prior-preservation is used to avoid overfitting and language-drift. Refer to the paper to learn more about it. For prior-preservation we first generate images using the model with a class prompt and then use those during training along with our data.

According to the paper, it's recommended to generate `num_epochs * num_samples` images for prior-preservation. 200-300 works well for most cases. The `num_class_images` flag sets the number of images to generate with the class prompt. You can place existing images in `class_data_dir`, and the training script will generate any additional images so that `num_class_images` are present in `class_data_dir` during training time. The general script can be then modified as the following.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

torchrun --nproc_per_node 2 train_dreambooth_colossalai.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --placement="cuda"
```

## Inference

Once you have trained a model using above command, the inference can be done simply using the `StableDiffusionPipeline`. Make sure to include the `identifier`(e.g. `--instance_prompt="a photo of sks dog" ` in the above example) in your prompt.

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-save-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```

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
