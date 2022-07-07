import torch
import torch.distributed as dist
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import LOSSES
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.modules.loss import _Loss


class _VocabParallelCrossEntropy1D(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, vocab_parallel_logits, targets, process_group):
        if process_group is None:
            process_group = gpc.get_group(ParallelMode.PARALLEL_1D)

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=process_group)
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = dist.get_rank(process_group)
        vocab_start_index = partition_vocab_size * rank
        vocab_end_index = vocab_start_index + partition_vocab_size

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (targets < vocab_start_index) | (targets >= vocab_end_index)
        masked_target = targets.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(targets)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=process_group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=process_group)

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits
        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


@LOSSES.register_module
class VocabParallelCrossEntropyLoss1D(_Loss):
    """Vocab parallel cross entropy loss for 1D parallelism.

    Args:
        reduction (bool, optional): whether to average the loss, defaults to True.
    """

    def __init__(self, reduction=True):
        super().__init__()
        self.reduction_mean = reduction

    def forward(self, logits, targets, process_group=None):
        """Calculate loss between logits and targets.

        Args:
            logits (:class:`torch.tensor`): Predicted unnormalized scores (often referred to as logits).
            targets (:class:`torch.tensor`): Ground truth class indices or class probabilities.
        """
        loss = _VocabParallelCrossEntropy1D.apply(logits, targets, process_group)
        if self.reduction_mean:
            loss = loss.mean()
        return loss
