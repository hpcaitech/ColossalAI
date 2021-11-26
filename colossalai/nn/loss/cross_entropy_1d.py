#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from colossalai.nn.layer.parallel_1d.layers import Linear1D_Col
from colossalai.utils.cuda import get_current_device
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from colossalai.registry import LOSSES

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_1d._utils import vocab_range_from_per_partition_vocab_size, vocab_range_from_global_vocab_size


class _VocabParallelCrossEntropy_1D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        # print("logits_max shape:", logits_max.size())
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=gpc.get_group(ParallelMode.PARALLEL_1D))
        # print("logits_max shape after all reduce:", logits_max.size())                             
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
        # vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))
        # print("vocab_parallel_logits shape:", vocab_parallel_logits.size())

        # Get the partition's vocab indecies
        # partition_vocab_size = vocab_parallel_logits.size()[-1]
        partition_vocab_size = vocab_parallel_logits.size(-1)
        rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
        world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
        vocab_start_index, vocab_end_index = vocab_range_from_global_vocab_size(
            partition_vocab_size, rank, world_size)
        # print("partition_vocab_size, rank, world_size, vocab_start_index, vocab_end_index: ", partition_vocab_size, rank, world_size, vocab_start_index, vocab_end_index)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        # logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size).contiguous()
        # masked_target_1d = masked_target.view(-1).contiguous()
        # arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
        #                          device=logits_2d.device)
        arange_1d = torch.arange(start=0, end=vocab_parallel_logits.size()[0])
        # predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = vocab_parallel_logits[arange_1d, masked_target]
        # predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        # predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Sum of exponential of logits along vocab dimension across all GPUs.
        # exp_logits = vocab_parallel_logits
        # torch.exp(vocab_parallel_logits, out=exp_logits)
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=gpc.get_group(ParallelMode.PARALLEL_1D))

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=get_current_device())
        grad_2d[arange_1d, masked_target_1d] -= (
                1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None

@LOSSES.register_module
class LmLoss1D(_Loss):

    def forward(self, lm_logits, lm_labels, loss_mask):
        lm_loss = _VocabParallelCrossEntropy_1D.apply(lm_logits, lm_labels)
        lm_loss = torch.sum(
            lm_loss.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        return lm_loss

@LOSSES.register_module
class SopLoss1D(_Loss):

    def forward(self, sop_logits, sentence_order):
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        return sop_loss

@LOSSES.register_module
class BERTDualHeadLoss(_Loss):

    def __init__(self):
        self.lm_loss = LmLoss1D()
        self.sop_loss = SopLoss1D()

    def forward(self, lm_logits, sop_logits, lm_labels, loss_mask, sentence_order):
        lm_loss = self.lm_loss(lm_logits, lm_labels, loss_mask)
        sop_loss = self.sop_loss(sop_logits, sentence_order)
        return lm_loss + sop_loss

@LOSSES.register_module
class CrossEntropyLoss1D(_Loss):
    """Cross entropy loss for 1D parallelism

    :param reduction: whether to average the loss, defaults to True
    :type reduction: bool, optional
    """

    def __init__(self):
        super().__init__()
        self.dim = gpc.tensor_parallel_size

    def forward(self, logits, targets):
        # loss = _VocabParallelCrossEntropy_1D.apply(
        #     logits, targets,
        # )
        # print("loss :", loss.size())
        # print("loss contiguous or not: ", loss.is_contiguous())
        # return loss

        # dist_loss = loss.mean()

        # test tp=1
        loss = torch.nn.CrossEntropyLoss()
        dist_loss = loss(logits, targets)
        return dist_loss
