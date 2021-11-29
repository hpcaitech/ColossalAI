import torch
import torch.distributed as dist
from torch.nn.modules.loss import _Loss

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_2d._utils import assert_summa_initialization, get_summa_dim_from_env
from colossalai.registry import LOSSES
from colossalai.utils import get_current_device
from torch.cuda.amp import custom_bwd, custom_fwd


class _ParallelCrossEntropyLossFunction_2D(torch.autograd.Function):
    ### Modified based on megatron.mpu.cross_entropy ###

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, targets):
        # logits: [b/q, h/q]
        # labels: [b/q]

        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
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
            start=0, end=logits.size()[0],
        )
        predicted_logits = logits[arange_1d, masked_target]
        predicted_logits[target_mask] = 0.
        dist.all_reduce(predicted_logits, group=gpc.get_group(
            ParallelMode.PARALLEL_2D_ROW))

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1)
        dist.all_reduce(sum_exp_logits, group=gpc.get_group(
            ParallelMode.PARALLEL_2D_ROW))

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
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=get_current_device())
        grad_2d[arange_1d,
                masked_target] -= (1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(output_grad.unsqueeze(dim=-1))

        return grad_input, None


class _ReduceByColumn(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        dist.all_reduce(input_, group=gpc.get_group(
            ParallelMode.PARALLEL_2D_COL))
        return input_

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input_):
        dist.all_reduce(input_, group=gpc.get_group(
            ParallelMode.PARALLEL_2D_COL))
        return input_

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output


@LOSSES.register_module
class CrossEntropyLoss2D(_Loss):
    """Cross entropy loss for 2D parallelism

    :param reduction: whether to average the loss, defaults to True
    :type reduction: bool, optional
    """

    def __init__(self, reduction=True):
        super().__init__()
        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
        self.reduction_mean = reduction

    def forward(self, logits, targets):
        targets = targets.chunk(self.summa_dim, dim=0)[self.row_rank]
        loss = _ParallelCrossEntropyLossFunction_2D.apply(
            logits, targets,
        )
        if self.reduction_mean:
            loss = _ReduceByColumn.apply(loss) / self.summa_dim
        dist_loss = loss.mean()

        return dist_loss
