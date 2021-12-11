# Overview

Here is an example of training ViT-B/16 on Imagenet-1K with batch size 32K.
We use 8x NVIDIA A100 GPU in this example. 

# How to run
Using [Slurm](https://slurm.schedmd.com/documentation.html):
```shell
srun python train_dali.py --local_rank=$SLURM_PROCID --world_size=$SLURM_NPROCS --host=$HOST --port=29500 --config=vit-b16.py
```

# Results

![Loss Curve](./loss.jpeg)
![Accuracy](./acc.jpeg)

# Details
`vit-b16.py`

It is a [config file](https://colossalai.org/config.html), which is used by ColossalAI to define all kinds of training arguments, such as the model, dataset, and training method (optimizer, lr_scheduler, epoch, etc.). You can access config content by `gpc.config`.

In this example, we train the ViT-Base patch 16 model 300 epochs on ImageNet-1K. The batch size is set to 32K through data parallel (4K on each GPU from 16x gradient accumulation with batch size 256). Since the batch size is very large than common usage, leading to convergence difficulties, we use a 
large batch optimizer [LAMB](https://arxiv.org/abs/1904.00962), and we can scale the batch size to 32K with a little accuracy loss. The learning rate and weight decay of the optimizer are set to 1.8e-2 and 0.1, respectively. We use a linear warmup learning rate scheduler and warmup 150 epochs.
We introduce FP16 mixed precision to accelerate training and use gradient clipping to help convergence.
For simplicity and speed, we didn't apply `RandAug` and just used [Mixup](https://arxiv.org/abs/1710.09412) in data augmentation.

If you have enough computing resources, you can expand this example conveniently with data parallel on a very large scale without gradient accumulation, and finish the training process even within one hour.


`imagenet_dali_dataloader.py`
To accelerate the training process, we use [DALI](https://github.com/NVIDIA/DALI) as data loader. Note that it requires the dataset in TFRecord format, avoiding read raw images which reduces efficiency of the file system.

`train_dali.py`
We build the DALI data loader and train process using Colossal-AI here.

`mixup.py`
Since we used Mixup, we define mixup loss in this file.

`hooks.py`
We also define useful hooks to log information help debugging.
