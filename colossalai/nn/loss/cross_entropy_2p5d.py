import torch
import torch.distributed as dist
from torch.nn.modules.loss import _Loss

from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.layer.parallel_2p5d._utils import assert_tesseract_initialization, \
    get_tesseract_dim_dep_from_env
from colossalai.registry import LOSSES
from colossalai.utils import get_current_device


class _ParallelCrossEntropyLossFunction_2p5D(torch.autograd.Function):
    ### Modified based on megatron.mpu.cross_entropy ###

    @staticmethod
    def forward(ctx, logits, targets):
        # logits: [b/dq, h/q]
        # loss: [b/dq]
        # targets: [b/dq, h/q]
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))
        # Subtract the maximum value.
        logits = logits - logits_max.unsqueeze(dim=-1)

        vocab_size = logits.size(-1)
        rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
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
        dist.all_reduce(predicted_logits, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1)
        dist.all_reduce(sum_exp_logits, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))

        loss = torch.log(sum_exp_logits) - predicted_logits

        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target)

        return loss

    @staticmethod
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


class _ReduceByColDep(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        dist.all_reduce(input_, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_XZ))
        return input_

    @staticmethod
    def forward(ctx, input_):
        dist.all_reduce(input_, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_XZ))
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


@LOSSES.register_module
class CrossEntropyLoss2p5D(_Loss):
    """Cross entropy loss for 2.5D parallelism

    :param reduction: whether to average the loss, defaults to True
    :type reduction: bool, optional
    """

    def __init__(self, reduction=True):
        super().__init__()
        assert_tesseract_initialization()
        self.xz_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_XZ)
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()
        self.reduction_mean = reduction

    def forward(self, logits, targets):
        targets = targets.chunk(self.tesseract_dim *
                                self.tesseract_dep, dim=0)[self.xz_rank]
        loss = _ParallelCrossEntropyLossFunction_2p5D.apply(
            logits, targets,
        )
        if self.reduction_mean:
            loss = _ReduceByColDep.apply(
                loss) / self.tesseract_dim / self.tesseract_dep
        dist_loss = loss.mean()

        return dist_loss
