# Vision Transformer with ColoTensor

# Overview

In this example, we will run Vision Transformer with ColoTensor.

We use model **ViTForImageClassification** from Hugging Face [Link](https://huggingface.co/docs/transformers/model_doc/vit) for unit test.
You can change world size or decide whether use DDP in our code.

We use model **vision_transformer** from timm [Link](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) for training example.

(2022/6/28) The default configuration now supports 2DP+2TP with gradient accumulation and checkpoint support. Zero is not supported at present.

# Requirement

Install colossalai version >= 0.1.11

## Unit test
To run unit test, you should install pytest, transformers with:
```shell
pip install pytest transformers
```

## Training example
To run training example with ViT-S, you should install **NVIDIA DALI** from [Link](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html) for dataloader support.
You also need to install timm and titans for model/dataloader support with:
```shell
pip install timm titans
```

### Data preparation
You can download the ImageNet dataset from the [ImageNet official website](https://www.image-net.org/download.php). You should get the raw images after downloading the dataset. As we use **NVIDIA DALI** to read data, we use the TFRecords dataset instead of raw Imagenet dataset. This offers better speedup to IO. If you don't have TFRecords dataset, follow [imagenet-tools](https://github.com/ver217/imagenet-tools) to build one.

Before you start training, you need to set the environment variable `DATA` so that the script knows where to fetch the data for DALI dataloader.
```shell
export DATA=/path/to/ILSVRC2012
```


# How to run

## Unit test
In your terminal
```shell
pytest test_vit.py
```

This will evaluate models with different **world_size** and **use_ddp**.

## Training example
Modify the settings in run.sh according to your environment.
For example, if you set `--nproc_per_node=8` in `run.sh` and `TP_WORLD_SIZE=2` in your config file,
data parallel size will be automatically calculated as 4.
Thus, the parallel strategy is set to 4DP+2TP.

Then in your terminal
```shell
sh run.sh
```

This will start ViT-S training with ImageNet.
