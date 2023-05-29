from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DistCrossEntropy(Function):
    """
    Overwrite the forward and backward function to calculate the cross entropy loss before gather

    Args:
        Function (:class:`torch.autograd.Function`): default
    """

    @staticmethod
    def forward(ctx, vocab_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        r"""
        Calculate the cross entropy loss before gather

        Args:
            vocab_logits (:class:`torch.Tensor`): The logits of the vocabulary, shape is
              [batch_size, seq_len, vocab_size]
            labels (:class:`torch.Tensor`): The labels of the vocabulary, shape is
              [batch_size, seq_len]

        Returns:
            :class:`torch.Tensor`: The cross entropy loss
        """
        partion_vocab_size = vocab_logits.shape[-1]
        ctx.vocab_size = partion_vocab_size * dist.get_world_size()

        #get the mask to filter the labels not in local device
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        delta = (partion_vocab_size * dist.get_world_size() + world_size - 1) // world_size
        down_shreshold = rank * delta
        up_shreshold = down_shreshold + delta
        # [down, up) => false, other => true
        mask = (labels < down_shreshold) | (labels >= up_shreshold)
        mask_labels = labels.clone() - down_shreshold
        # the default ignore index is -100
        mask_labels[mask] = -100

        # reshape the vocab_logits to [bath_size * seq_len, vocab_size]
        # reshape the labels to [bath_size * seq_len]
        vocab_logits_2d = vocab_logits.view(-1, partion_vocab_size)
        labels_1d = mask_labels.view(-1)

        exp_vocab_logits_2d = torch.exp(vocab_logits_2d)
        sum_exp_vocab_logits_2d = torch.sum(exp_vocab_logits_2d, dim=-1)
        dist.all_reduce(sum_exp_vocab_logits_2d, op=dist.ReduceOp.SUM)

        log_softmax_vocab_logits_2d = torch.log(exp_vocab_logits_2d / sum_exp_vocab_logits_2d.unsqueeze(-1))
        loss = F.nll_loss(log_softmax_vocab_logits_2d, labels_1d, reduction="none")    # the ignore index is -100
        loss_list = [torch.empty_like(loss) for _ in range(world_size)]
        loss_list[rank] = loss
        dist.all_gather(loss_list, loss)
        loss = torch.cat(loss_list, dim=0)
        non_zero_count = torch.sum(loss != 0)
        loss = loss.sum() / non_zero_count

        log_softmax_vocab_logits = log_softmax_vocab_logits_2d.view(*vocab_logits.shape)
        ctx.save_for_backward(log_softmax_vocab_logits, mask, labels_1d)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # retrieve the saved tensors and set the ignore to 0 to avoid out out index
        log_softmax_vocab_logits, mask, labels_1d = ctx.saved_tensors
        labels_1d[labels_1d == -100] = 0

        # logsoftmax as the grad_input
        grad_input = log_softmax_vocab_logits
        partion_vocab_size = log_softmax_vocab_logits.shape[-1]
        grad_2d = grad_input.view(-1, partion_vocab_size)

        # set a mask to update the gradient of the labels in local device
        arange_1d = torch.arange(start=0, end=grad_2d.shape[0], device=grad_2d.device)
        logsoftmax_update = 1.0 - mask.view(-1).float()
        grad_2d[arange_1d, labels_1d] -= logsoftmax_update

        # calculate the grad_input
        grad_input.mul_(grad_outputs.unsqueeze(-1))

        return grad_input, None, None


def applyDistCrossEntropy(vocab_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return DistCrossEntropy.apply(vocab_logits, labels)
