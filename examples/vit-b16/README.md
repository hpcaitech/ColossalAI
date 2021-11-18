# Overview

Here is an example of training ViT-B/16 on Imagenet-1K. We use 8x A100 in this example. For simplicity and speed, we didn't apply `RandAug` and we just used `Mixup`. With `LAMB` optimizer, we can scale the batch size to 32K with a little accuracy loss.

# How to run
Using slurm:
```shell
srun python train_dali.py --local_rank=$SLURM_PROCID --world_size=$SLURM_NPROCS --host=$HOST --port=29500 --config=vit-b16.py
```

# Results

![Loss Curve](./loss.jpeg)
![Accuracy](./acc.jpeg)
