import torch
import torch.distributed as dist
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _Loss

from colossalai.accelerator import get_accelerator
from colossalai.legacy.constants import INPUT_GROUP_3D, OUTPUT_GROUP_3D, WEIGHT_GROUP_3D
from colossalai.legacy.core import global_context as gpc
from colossalai.legacy.nn.layer.parallel_3d import reduce_by_batch_3d, split_tensor_3d
from colossalai.legacy.nn.layer.parallel_3d._utils import get_parallel_mode_from_env
from colossalai.legacy.registry import LOSSES


@LOSSES.register_module
class CrossEntropyLoss3D(_Loss):
    r"""Cross entropy loss for 3D parallelism.

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
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.reduction_mean = reduction
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits, targets):
        """Calculate loss between logits and targets.

        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_tensor_3d(targets, 0, self.weight_parallel_mode)
        targets = split_tensor_3d(targets, 0, self.input_parallel_mode)
        loss = cross_entropy(logits, targets, reduction="none", *self.loss_args, **self.loss_kwargs)
        if self.reduction_mean:
            loss = loss.mean()
            loss = reduce_by_batch_3d(loss, self.input_parallel_mode, self.weight_parallel_mode, True)
        return loss


class _VocabParallelCrossEntropy3D(torch.autograd.Function):
    # Adapted from megatron.mpu.cross_entropy
    # loss[i] = -logits[i][targets] + log(sum(exp(logits[i])))

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, targets, output_parallel_mode):
        # logits: [b/q^2, c/q]
        # labels: [b/q^2]
        # loss: [b/q^2]
        logits_max = torch.max(logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=gpc.get_group(output_parallel_mode))
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
        arange_1d = torch.arange(start=0, end=logits.size()[0], device=get_accelerator().get_current_device())
        predicted_logits = logits[arange_1d, masked_target]
        predicted_logits = predicted_logits.clone().contiguous().view_as(targets)
        predicted_logits[target_mask] = 0.0
        dist.all_reduce(predicted_logits, group=gpc.get_group(output_parallel_mode))

        # Loss = log(sum(exp(logits))) - predicted-logit.
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, group=gpc.get_group(output_parallel_mode))
        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target)

        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        # Retrieve tensors from the forward path.
        softmax, target_mask, masked_target = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        input_grad = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = input_grad.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=get_accelerator().get_current_device())
        grad_2d[arange_1d, masked_target] -= 1.0 - target_mask.view(-1).float()
        input_grad.mul_(output_grad.unsqueeze(dim=-1))

        return input_grad, None, None, None


@LOSSES.register_module
class VocabParallelCrossEntropyLoss3D(_Loss):
    """Vocab parallel cross entropy loss for 2D parallelism.

    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(self, reduction=True):
        super().__init__()
        self.input_parallel_mode = get_parallel_mode_from_env(INPUT_GROUP_3D)
        self.weight_parallel_mode = get_parallel_mode_from_env(WEIGHT_GROUP_3D)
        self.output_parallel_mode = get_parallel_mode_from_env(OUTPUT_GROUP_3D)
        self.reduction_mean = reduction

    def forward(self, logits, targets):
        """Calculate loss between logits and targets.

        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        targets = split_tensor_3d(targets, 0, self.weight_parallel_mode)
        targets = split_tensor_3d(targets, 0, self.input_parallel_mode)
        loss = _VocabParallelCrossEntropy3D.apply(logits, targets, self.output_parallel_mode)
        if self.reduction_mean:
            loss = loss.mean()
            loss = reduce_by_batch_3d(loss, self.input_parallel_mode, self.weight_parallel_mode, True)
        return loss
