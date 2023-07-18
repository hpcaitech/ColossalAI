# Gradient Accumulation (Outdated)

Author: Shenggui Li, Yongbin Li

**Prerequisite**
- [Define Your Configuration](../basics/define_your_config.md)
- [Use Engine and Trainer in Training](../basics/engine_trainer.md)

**Example Code**
- [ColossalAI-Examples Gradient Accumulation](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_accumulation)

## Introduction

Gradient accumulation is a common way to enlarge your batch size for training.
When training large-scale models, memory can easily become the bottleneck and the batch size can be very small, (e.g. 2),
leading to unsatisfactory convergence. Gradient accumulation works by adding up the gradients calculated in multiple iterations,
and only update the parameters in the preset iteration.

## Usage

It is simple to use gradient accumulation in Colossal-AI. Just add this following configuration into your config file.
The integer represents the number of iterations to accumulate gradients.

```python
gradient_accumulation = <int>
```

## Hands-on Practice

We provide a [runnable example](https://github.com/hpcaitech/ColossalAI-Examples/tree/main/features/gradient_accumulation)
to demonstrate gradient accumulation. In this example, we set the gradient accumulation size to be 4. You can run the script using this command:

```shell
python -m torch.distributed.launch --nproc_per_node 1 --master_addr localhost --master_port 29500  run_resnet_cifar10_with_engine.py
```

You will see output similar to the text below. This shows gradient is indeed accumulated as the parameter is not updated
in the first 3 steps, but only updated in the last step.

```text
iteration 0, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 1, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 2, first 10 elements of param: tensor([-0.0208,  0.0189,  0.0234,  0.0047,  0.0116, -0.0283,  0.0071, -0.0359, -0.0267, -0.0006], device='cuda:0', grad_fn=<SliceBackward0>)
iteration 3, first 10 elements of param: tensor([-0.0141,  0.0464,  0.0507,  0.0321,  0.0356, -0.0150,  0.0172, -0.0118, 0.0222,  0.0473], device='cuda:0', grad_fn=<SliceBackward0>)
```

<!-- doc-test-command: torchrun --standalone --nproc_per_node=1 gradient_accumulation.py  -->
