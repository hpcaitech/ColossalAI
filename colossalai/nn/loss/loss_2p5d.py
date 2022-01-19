from colossalai.nn.layer.parallel_2p5d import reduce_by_batch_2p5d
from colossalai.nn.layer.parallel_2p5d._utils import assert_tesseract_initialization
from colossalai.registry import LOSSES
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _Loss


@LOSSES.register_module
class CrossEntropyLoss2p5D(_Loss):
    """
    Cross entropy loss for 2.5D parallelism
    
    :param reduction: whether to average the loss, defaults to True
    :param args: Args for loss function
    :param kwargs: Kwargs for loss function

    :type reduction: bool, optional
    """
    def __init__(self, reduction=True, *args, **kwargs):
        super().__init__()
        assert_tesseract_initialization()
        self.reduction_mean = reduction
        self.loss_args = args
        self.loss_kwargs = kwargs

    def forward(self, logits, targets):
        """Calculate loss between logits and targets

        :param logits: Output logits of model
        :param targets: True targets from data
        """
        loss = cross_entropy(logits, targets, reduction='none', *self.loss_args, **self.loss_kwargs)
        if self.reduction_mean:
            loss = loss.mean()
            loss = reduce_by_batch_2p5d.apply(loss, True)
        return loss
