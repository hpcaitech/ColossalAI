import torch
import torch.distributed as dist
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_2d import reduce_by_batch_2d, split_batch_2d
from colossalai.nn.layer.parallel_2d._utils import assert_summa_initialization
from colossalai.registry import LOSSES
from colossalai.utils import get_current_device
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _Loss


@LOSSES.register_module
class CrossEntropyLoss2D(_Loss):
    r"""Cross entropy loss for 2D parallelism

    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.

    The ``args`` and ``kwargs`` should include parameters below:
    ::

        weight (Tensor, optional)
        size_average (bool, optional)
        ignore_index (int, optional)
        reduce (bool, optional)
        label_smoothing (float, optional)

    More details about ``args``, ``kwargs`` and ``torch.nn.functional.cross_entropy`` could be found in
    `Cross_entropy <https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy>`_.
    """

    def __init__(self, reduction=True, *args, **kwargs):
        super().__init__()
        assert_summa_initialization()
        self.reduction_mean = reduction
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits, targets):
        """Calculate loss between logits and targets.

        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.

        Returns:
            float: the loss between logits and targets.
        """
        targets = split_batch_2d(targets)
        loss = cross_entropy(logits, targets, reduction='none', *self.loss_args, **self.loss_kwargs)
        if self.reduction_mean:
            loss = loss.mean()
            loss = reduce_by_batch_2d(loss, True)
        return loss


class _VocabParallelCrossEntropy2D(torch.autograd.Function):
    ### Modified based on megatron.mpu.cross_entropy ###

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, targets):
        # logits: [b/q, h/q]
        # labels: [b/q]
        # loss: [b/q]
        # vocab_parallel_logits: [b/q, s, v/q]
        # target: [b/q, s]
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=gpc.get_group(ParallelMode.PARALLEL_2D_ROW))
        # Subtract the maximum value.
        # vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))
        logits = logits - logits_max.unsqueeze(dim=-1)

        vocab_size = logits.size(-1)
        rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)
        vocab_start = rank * (vocab_size)
        vocab_end = (rank + 1) * (vocab_size) - 1

        target_mask = (targets < vocab_start) | (targets > vocab_end)

        masked_target = targets.clone() - vocab_start
        masked_target[target_mask] = 0
        arange_1d = torch.arange(
            start=0,
            end=logits.size()[0],
        )
        predicted_logits = logits[arange_1d, masked_target]
        predicted_logits[target_mask] = 0.
        dist.all_reduce(predicted_logits, group=gpc.get_group(ParallelMode.PARALLEL_2D_ROW))

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1)
        dist.all_reduce(sum_exp_logits, group=gpc.get_group(ParallelMode.PARALLEL_2D_ROW))

        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target)

        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=get_current_device())
        grad_2d[arange_1d, masked_target] -= (1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(output_grad.unsqueeze(dim=-1))

        return grad_input, None


@LOSSES.register_module
class VocabParallelCrossEntropyLoss2D(_Loss):
    """Vocab parallel cross entropy loss for 2D parallelism.

    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(self, reduction=True):
        super().__init__()
        self.reduction_mean = reduction

    def forward(self, logits, targets):
        """Calculate loss between logits and targets.

        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_batch_2d(targets)
        loss = _VocabParallelCrossEntropy2D.apply(
            logits,
            targets,
        )
        if self.reduction_mean:
            loss = loss.mean()
            loss = reduce_by_batch_2d(loss, True)
        return loss
