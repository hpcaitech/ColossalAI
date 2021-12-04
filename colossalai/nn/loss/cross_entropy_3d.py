#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

import torch
import torch.distributed as dist
from colossalai.constants import (INPUT_GROUP_3D, OUTPUT_GROUP_3D,
                                  WEIGHT_GROUP_3D)
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_3d._operation import Reduce_3D
from colossalai.nn.layer.parallel_3d._utils import (get_depth_from_env,
                                                    get_last_group,
                                                    get_parallel_mode_from_env)
from colossalai.registry import LOSSES
from colossalai.utils import get_current_device
from torch.nn.modules.loss import _Loss


class _ParallelCrossEntropyLossFunction_3D(torch.autograd.Function):
    """
    Adapted from megatron.mpu.cross_entropy
    loss[i] = -logits[i][targets] + log(sum(exp(logits[i])))
    """
    @staticmethod
    def forward(ctx, logits, targets, depth, output_parallel_mode):
        # logits: [b/q^2, c/q]
        # labels: [b/q^2]
        # loss: [b/q^2]
        logits_max = torch.max(logits, dim=-1)[0]
        dist.all_reduce(logits_max,
                        op=torch.distributed.ReduceOp.MAX,
                        group=gpc.get_group(output_parallel_mode))
        # Subtract the maximum value.
        logits = logits - logits_max.unsqueeze(dim=-1)

        vocab_size_per_partition = logits.size()[-1]
        rank = gpc.get_local_rank(output_parallel_mode)
        vocab_start = rank * vocab_size_per_partition
        vocab_end = (rank + 1) * vocab_size_per_partition - 1

        # loss[i] = 0 if targets[i] < vocab_start or targets[i] > vocab_end
        target_mask = (targets < vocab_start) | (targets > vocab_end)
        masked_target = targets.clone() - vocab_start
        masked_target[target_mask] = 0
        arange_1d = torch.arange(start=0,
                                 end=logits.size()[0],
                                 device=get_current_device())
        predicted_logits = logits[arange_1d, masked_target]
        predicted_logits = predicted_logits.clone().contiguous().view_as(
            targets)
        predicted_logits[target_mask] = 0.
        dist.all_reduce(predicted_logits,
                        group=gpc.get_group(output_parallel_mode))

        # Loss = log(sum(exp(logits))) - predicted-logit.
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits,
                        group=gpc.get_group(output_parallel_mode))
        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target)

        return loss

    @staticmethod
    def backward(ctx, output_grad):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        input_grad = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = input_grad.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0,
                                 end=grad_2d.size()[0],
                                 device=get_current_device())
        grad_2d[arange_1d,
                masked_target] -= (1.0 - target_mask.view(-1).float())
        input_grad.mul_(output_grad.unsqueeze(dim=-1))

        return input_grad, None, None, None


@LOSSES.register_module
class CrossEntropyLoss3D(_Loss):
    """Cross entropy loss for 3D parallelism

    :param depth: depth for 3D parallelism
    :type depth: int
    :param input_parallel_mode: parallel mode for input tensor
    :type input_parallel_mode: ParallelMode
    :param weight_parallel_mode: parallel mode for weight
    :type weight_parallel_mode: ParallelMode
    :param reduction: whether to average the loss, defaults to True
    :type reduction: bool, optional
    """
    def __init__(
            self,
            #  input_parallel_mode,
            #  weight_parallel_mode,
            reduction=True,
            label_smoothing=0.0):
        super().__init__()
        self.depth = get_depth_from_env()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_last_group(self.input_parallel_mode,
                                                   self.weight_parallel_mode)
        self.input_rank = gpc.get_local_rank(self.input_parallel_mode)
        self.weight_rank = gpc.get_local_rank(self.weight_parallel_mode)
        self.reduction_mean = reduction

    def forward(self, logits, targets):
        # split label partition from the entire batch
        batch_size = targets.size(0)
        targets = torch.chunk(targets, self.depth, dim=0)[self.weight_rank]
        targets = torch.chunk(targets, self.depth, dim=0)[self.input_rank]
        loss = _ParallelCrossEntropyLossFunction_3D.apply(
            logits, targets, self.depth, self.output_parallel_mode)
        if self.reduction_mean:
            loss = loss.sum()
            loss = Reduce_3D.apply(loss, self.depth, self.input_parallel_mode)
            loss = Reduce_3D.apply(loss, self.depth, self.weight_parallel_mode)
            loss /= batch_size
        return loss


# @LOSSES.register_module
# class LabelSmoothingCrossEntropy3D(_Loss):
#     """
#     NLL loss with label smoothing, adapted from timm.loss.LabelSmoothingCrossEntropy

#     :param input_parallel_mode: parallel mode for input tensor
#     :type input_parallel_mode: ParallelMode
#     :param weight_parallel_mode: parallel mode for weight
#     :type weight_parallel_mode: ParallelMode
#     :param smoothing: label smoothing value, defaults to 0.1
#     :type smoothing: float
#     :param reduction: whether to average the loss, defaults to True
#     :type reduction: bool, optional
#     """
#     def __init__(self,
#                  input_parallel_mode,
#                  weight_parallel_mode,
#                  smoothing=0.1,
#                  reduction=True):
#         super().__init__()
#         assert smoothing < 1.0
#         self.smoothing = smoothing
#         self.confidence = 1. - smoothing
#         self.depth = get_depth_from_env()
#         self.input_parallel_mode = input_parallel_mode
#         self.weight_parallel_mode = weight_parallel_mode
#         self.output_parallel_mode = get_last_group(input_parallel_mode,
#                                                    weight_parallel_mode)
#         self.reduction_mean = reduction

#     def forward(self, logits, targets):
#         # split label partition from the entire batch
#         j = gpc.get_local_rank(self.input_parallel_mode)
#         i = gpc.get_local_rank(self.weight_parallel_mode)
#         targets = torch.chunk(targets, self.depth, dim=0)[i]
#         targets = torch.chunk(targets, self.depth, dim=0)[j]
#         exp_logits = torch.exp(logits)
#         sum_exp_logits = Sum3D.apply(exp_logits, -1, depth,
#                                      self.output_parallel_mode, False)
#         log_probs = torch.log(sum_exp_logits) - logits
#         nll_loss = _ParallelCrossEntropyLossFunction_3D.apply(
#             logits, targets, self.depth, self.output_parallel_mode)
#         smooth_loss = -log_probs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         if self.reduction_mean:
#             loss = loss.sum()
#             loss = Reduce_3D.apply(loss, self.depth, self.input_parallel_mode)
#             loss = Reduce_3D.apply(loss, self.depth, self.weight_parallel_mode)
#             loss /= batch_size
#         return loss
