#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import distributed as dist
from torch.cuda.amp import custom_bwd, custom_fwd

from colossalai.accelerator import get_accelerator
from colossalai.legacy.communication import ring_forward
from colossalai.legacy.context.parallel_mode import ParallelMode
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn.layer.parallel_sequence._utils import _calc_current_device_range, _calc_incoming_device_range


class RingQK(torch.autograd.Function):
    """
    Calculate QK in a ring-exchange style
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, sub_q, sub_k, batch_size, num_attention_heads, sub_seq_length):
        # save tensor for backward
        ctx.save_for_backward(sub_q, sub_k)
        ctx.sub_seq_length = sub_seq_length

        # create local segment of attention score
        attention_score = torch.empty(
            batch_size * num_attention_heads,
            sub_seq_length,
            sub_seq_length * gpc.get_world_size(ParallelMode.SEQUENCE),
            dtype=sub_q.dtype,
            device=get_accelerator().get_current_device(),
        )

        # compute local QK^T
        part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        start_idx = local_rank * sub_seq_length
        end_idx = (local_rank + 1) * sub_seq_length
        attention_score[:, :, start_idx:end_idx] = part_a

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_k = ring_forward(sub_k, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)
            part_a = torch.matmul(sub_q, sub_k.transpose(2, 1))
            attention_score[:, :, start_idx:end_idx] = part_a

        return attention_score

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        (
            sub_q,
            sub_k,
        ) = ctx.saved_tensors
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)

        # calculate gradient of sub_k
        grad_k = torch.matmul(grad_output.transpose(2, 1), sub_q)

        dist.all_reduce(grad_k, group=gpc.get_group(ParallelMode.SEQUENCE))
        grad_k = grad_k[:, local_rank * ctx.sub_seq_length : (local_rank + 1) * ctx.sub_seq_length]
        grad_k /= local_world_size

        # calculate gradient for sub_q
        grad_q = torch.zeros_like(
            sub_q,
            dtype=sub_q.dtype,
            device=get_accelerator().get_current_device(),
        )

        # compute with local sub_k
        start_idx, end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_length)
        grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_k = ring_forward(sub_k, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_length)
            grad_q += torch.matmul(grad_output[:, :, start_idx:end_idx], sub_k)

        grad_q /= local_world_size

        return grad_q, grad_k, None, None, None


class RingAV(torch.autograd.Function):
    """
    Calculate AV in a ring-exchange style
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, attention_score, sub_v, batch_size, num_attention_heads, attention_head_size, sub_seq_length):
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, sub_seq_length)

        sub_attention_result = torch.zeros(
            batch_size * num_attention_heads,
            sub_seq_length,
            attention_head_size,
            device=get_accelerator().get_current_device(),
            dtype=attention_score.dtype,
        )

        # save tensors for backward
        ctx.save_for_backward(attention_score, sub_v)
        ctx.sub_seq_length = sub_seq_length

        # compute local AV
        part_av = torch.matmul(attention_score[:, :, local_start_idx:local_end_idx], sub_v)
        sub_attention_result += part_av

        # compute AV in ring - all - reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_forward(sub_v, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, sub_seq_length)

            # compute QK^T
            part_av = torch.matmul(attention_score[:, :, start_idx:end_idx], sub_v)
            sub_attention_result += part_av
        return sub_attention_result

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        local_rank = gpc.get_local_rank(ParallelMode.SEQUENCE)
        local_world_size = gpc.get_world_size(ParallelMode.SEQUENCE)
        local_start_idx, local_end_idx = _calc_current_device_range(local_rank, ctx.sub_seq_length)
        attention_scores, sub_v = ctx.saved_tensors

        # calculate gradient of v
        grad_v = torch.matmul(attention_scores.transpose(2, 1), grad_output)
        dist.all_reduce(grad_v, group=gpc.get_group(ParallelMode.SEQUENCE))
        grad_v = grad_v[:, local_start_idx:local_end_idx]
        grad_v /= local_world_size

        # calculate gradient for attention score
        grad_attention_score = torch.zeros_like(
            attention_scores, dtype=grad_output.dtype, device=get_accelerator().get_current_device()
        )

        # compute with local sub_k
        grad_attention_score[:, :, local_start_idx:local_end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))

        # compute QK^T in ring-all-reduce style
        for i in range(local_world_size - 1):
            sub_v = ring_forward(sub_v, ParallelMode.SEQUENCE)
            start_idx, end_idx = _calc_incoming_device_range(i, local_rank, local_world_size, ctx.sub_seq_length)

            # compute grad_q
            grad_attention_score[:, :, start_idx:end_idx] += torch.matmul(grad_output, sub_v.transpose(2, 1))

        return grad_attention_score, grad_v, None, None, None, None
