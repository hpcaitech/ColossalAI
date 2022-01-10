from colossalai.nn.layer.parallel_2d import reduce_by_batch_2d
from colossalai.nn.layer.parallel_2d._utils import assert_summa_initialization
from colossalai.registry import LOSSES
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _Loss


@LOSSES.register_module
class CrossEntropyLoss2D(_Loss):
    """
    Cross entropy loss for 2D parallelism

    :param reduction: whether to average the loss, defaults to True
    :type reduction: bool, optional
    """
    def __init__(self, reduction=True, *args, **kwargs):
        super().__init__()
        assert_summa_initialization()
        self.reduction_mean = reduction
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits, targets):
        loss = cross_entropy(logits, targets, reduction='none', *self.loss_args, **self.loss_kwargs)
        if self.reduction_mean:
            loss = loss.mean()
            loss = reduce_by_batch_2d.apply(loss, True)
        return loss
