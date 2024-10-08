import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import ProcessGroup
from torch.nn import CrossEntropyLoss

from colossalai.shardformer.layer._operation import reduce_forward
from colossalai.shardformer.shard import ShardConfig

from .utils import is_share_sp_tp

__all__ = ["DistCrossEntropy", "cross_entropy_1d", "dist_cross_entropy"]

_IGNORE_IDX = -100


class DistCrossEntropy(Function):
    r"""
    Overwrite the forward and backward function to calculate the cross entropy loss before gather

    Args:
        Function (:class:`torch.autograd.Function`): default
    """

    @staticmethod
    def forward(
        ctx,
        vocab_logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int,
        process_group: ProcessGroup,
        vocab_size: int,
        dtype=torch.float32,
        mode="mean",
    ):
        r"""
        Calculate the cross entropy loss before gather, the origin loss function is as follows:
        loss = -log(exp(x[class])/sum(exp(x[i]))
        and can be rewriten as:
        loss = log(sum(exp(x[i])) - x[class]

        To avoid the `nan` of log(sum(exp(x[i]))), we minus the max of x[i]

        Args:
            vocab_logits (:class:`torch.Tensor`): The logits of the vocabulary, shape is
              [batch_size, seq_len, vocab_size]
            target (:class:`torch.Tensor`): The labels of the vocabulary, shape is
              [batch_size, seq_len]

        Returns:
            :class:`torch.Tensor`: The cross entropy loss
        """
        assert mode in ["mean", "sum"]
        # get the max
        logits_max = torch.max(vocab_logits, dim=-1)[0]
        handle = dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group, async_op=True)

        # mask the target in the local device
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        if vocab_size == None:
            partition_vocab_size = vocab_logits.size()[-1]
            global_vocab_size = partition_vocab_size * world_size
        else:
            global_vocab_size = vocab_size
            partition_vocab_size = global_vocab_size // world_size

        # [down, up) => false, other device and -100 => true
        delta = (global_vocab_size + world_size - 1) // world_size
        down_threshold = rank * delta
        up_threshold = down_threshold + delta
        if up_threshold > global_vocab_size:
            up_threshold = global_vocab_size
        mask = (target < down_threshold) | (target >= up_threshold)
        masked_target = target.clone() - down_threshold
        masked_target[mask] = 0
        masked_target_1d = masked_target.view(-1).contiguous()

        # minus the max to avoid the result of sum of exp is too large and the log is nan
        handle.wait()
        vocab_logits = vocab_logits - logits_max.unsqueeze(dim=-1)
        # reshape the logits and target
        # reshape the vocab_logits to [bath_size * seq_len, vocab_size]
        # reshape the labels to [bath_size * seq_len]
        self_vocab_size = vocab_logits.size()[-1]
        logits_2d = vocab_logits.view(-1, self_vocab_size)

        # extract the x[class] and set the x[other device] to zero
        idx = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        pred_logits_1d = logits_2d[idx, masked_target_1d].contiguous()
        pred_logits = pred_logits_1d.view_as(target)
        pred_logits[mask] = 0.0

        # all-reduce to get full x[i, y]
        handle = dist.all_reduce(pred_logits, op=dist.ReduceOp.SUM, group=process_group, async_op=True)
        exp_logits = vocab_logits
        torch.exp(vocab_logits, out=exp_logits)
        sum_exp_logits = torch.sum(exp_logits, dim=-1, dtype=torch.float32)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=process_group)

        # calculate the loss
        # loss = log(sum(exp(x[i]))) - x[class]
        handle.wait()
        loss = torch.where(target == ignore_index, 0.0, torch.log(sum_exp_logits) - pred_logits)
        if mode == "mean":
            num_non_zero = torch.sum(loss != 0.0)
            ctx.inv_num_non_zero = 1.0 / num_non_zero
            loss = torch.sum(loss).div_(num_non_zero)
        else:
            loss = torch.sum(loss)

        # calculate the softmax
        exp_logits = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1)).to(dtype)
        exp_logits[target == ignore_index] = 0.0
        ctx.save_for_backward(exp_logits, mask, masked_target_1d)
        ctx.dtype = dtype
        ctx.mode = mode

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve the saved tensors
        if ctx.mode == "mean":
            grad_output = grad_output * ctx.inv_num_non_zero
        exp_logits, mask, masked_target_1d = ctx.saved_tensors

        # use exp logits as the input grad
        grad_logits = exp_logits
        partion_vocab_size = grad_logits.shape[-1]
        grad_logits_2d = grad_logits.view(-1, partion_vocab_size)

        update = 1.0 - mask.view(-1).float().to(ctx.dtype)
        grad_logits_2d[torch.arange(0, grad_logits_2d.shape[0]), masked_target_1d] -= update

        grad_logits.mul_(grad_output.unsqueeze(dim=-1))
        return grad_logits, None, None, None, None, None, None


def cross_entropy_1d(
    vocab_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = _IGNORE_IDX,
    process_group: ProcessGroup = None,
    vocab_size: int = None,
    dtype: torch.dtype = None,
    mode: str = "mean",
) -> torch.Tensor:
    return DistCrossEntropy.apply(vocab_logits, labels, ignore_index, process_group, vocab_size, dtype, mode)


def dist_cross_entropy(
    labels: torch.Tensor,  # [B, S] or [B, S, Vocab_size]
    logits: torch.Tensor,  # [B, S, Vocab_size]
    shard_config: ShardConfig,
    vocab_size: int,
    dtype: torch.dtype,
    seq_dim: int = 1,
) -> torch.Tensor:
    """
    Helper to compute cross entropy loss for most shardformer models supporting PP, TP and SP.
    """
    # Split labels if not gather output
    sp_group = shard_config.sequence_parallel_process_group
    sp_rank = dist.get_rank(sp_group)
    sp_size = shard_config.sequence_parallel_size
    sp_mode = shard_config.sequence_parallelism_mode
    parallel_output = shard_config.parallel_output
    is_tp = shard_config.enable_tensor_parallelism
    is_packed = labels.dim() == 2
    if is_packed:
        bs, seq_len = labels.shape
    else:
        # padded sequence
        seq_len = labels.shape[-1]
        logits = logits.reshape(-1, *logits.shape[2:])
        seq_dim = 0

    # Shift labels to predict the next token, and remove the tail logit predicting <EOS>
    is_sp = sp_size > 1 and (not is_share_sp_tp(sp_mode))
    split_labels_here = seq_len // sp_size == logits.size(seq_dim)  # ring attn splits labels before forward

    if sp_mode == "ring_attn":
        # For Zigzag Ring Attention, labels should've been split and
        # shifted by RingAttention.prepare_varlen_batch()
        if sp_rank == 0:
            logits = logits[..., :-1, :]
            logits = torch.cat([logits, torch.full_like(logits[:, :1, :], _IGNORE_IDX)], dim=seq_dim)
    elif is_sp:
        # Shift only once: either before splitting or in the last rank without splitting
        if split_labels_here or (sp_rank == sp_size - 1):
            labels = labels[..., 1:]
        if split_labels_here:
            labels = labels.split(seq_len // sp_size, dim=-1)[sp_rank]

        if sp_rank == sp_size - 1:
            logits = logits[..., :-1, :]
            # Pad logits and labels to the same shape across all ranks for TP all_reduce
            if is_tp and parallel_output:
                # If is packed sequence (label dim is 1), then each seq already has the end label token padded.
                # torch.cat is faster than F.pad...
                pad_shape = (logits.shape[0], 1, *logits.shape[2:]) if is_packed else (1, *logits.shape[1:])
                padding = torch.full(pad_shape, _IGNORE_IDX, dtype=logits.dtype, device=logits.device)
                logits = torch.cat([logits, padding], dim=seq_dim)
                pad_shape = (labels.shape[0], 1) if is_packed else (1,)
                padding = torch.full(pad_shape, _IGNORE_IDX, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([labels, padding], dim=seq_dim)
    else:
        labels = labels[..., 1:]
        logits = logits[..., :-1, :]
    labels = labels.contiguous()
    logits = logits.contiguous()
    num_nonzero = (labels != _IGNORE_IDX).sum()
    assert labels.shape == logits.shape[:-1], f"label shape {labels.shape} does not match logit shape {logits.shape}"

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=_IGNORE_IDX, reduction="sum")
    labels = labels.view(-1)

    if is_tp and parallel_output:
        # Cross entropy with all-reduce for TP
        new_vocab_size = logits.shape[-1]
        logits = logits.view(-1, new_vocab_size)
        loss = cross_entropy_1d(
            logits,
            labels,
            process_group=shard_config.tensor_parallel_process_group,
            vocab_size=vocab_size,
            dtype=dtype,
            mode="sum",
        )
    else:
        # NOTE if use TP and not parallel_output, the output is gathered in VocabParallelLMHead1D
        logits = logits.view(-1, logits.size(-1))
        loss = loss_fct(logits, labels)

    # Reduce loss instead of gathering logits over seq dim for savings
    if split_labels_here or sp_mode == "ring_attn":
        # Get the global non-zero count
        loss = torch.stack((loss, num_nonzero))
        # Rescale to offset the grad / (DP * SP) in HybridParallelPlugin
        loss = reduce_forward(loss, sp_group, grad_scale=sp_size)
        loss, num_nonzero = loss[0], loss[1].detach()
    loss = (loss / num_nonzero).squeeze()
    return loss
