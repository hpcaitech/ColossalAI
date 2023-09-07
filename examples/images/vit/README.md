## Overview

Vision Transformer is a class of Transformer model tailored for computer vision tasks. It was first proposed in paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) and achieved SOTA results on various tasks at that time.

In our example, we are using pretrained weights of ViT loaded from HuggingFace.
We adapt the ViT training code to ColossalAI by leveraging [Boosting API](https://colossalai.org/docs/basics/booster_api) loaded with a chosen plugin, where each plugin corresponds to a specific kind of training strategy. This example supports plugins including TorchDDPPlugin (DDP), LowLevelZeroPlugin (Zero1/Zero2), GeminiPlugin (Gemini) and HybridParallelPlugin (any combination of tensor/pipeline/data parallel).

## Run Demo

By running the following script:
```bash
bash run_demo.sh
```
You will finetune a a [ViT-base](https://huggingface.co/google/vit-base-patch16-224) model on this [dataset](https://huggingface.co/datasets/beans), with more than 8000 images of bean leaves. This dataset is for image classification task and there are 3 labels: ['angular_leaf_spot', 'bean_rust', 'healthy'].

The script can be modified if you want to try another set of hyperparameters or change to another ViT model with different size.

The demo code refers to this [blog](https://huggingface.co/blog/fine-tune-vit).



## Run Benchmark

You can run benchmark for ViT model by running the following script:
```bash
bash run_benchmark.sh
```
The script will test performance (throughput & peak memory usage) for each combination of hyperparameters. You can also play with this script to configure your own set of hyperparameters for testing.
