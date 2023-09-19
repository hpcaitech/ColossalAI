import inspect
import math
from typing import Callable

from torch import dtype, nn

from colossalai.nn import init

from ..parallel_1d import *
from ..parallel_2d import *
from ..parallel_2p5d import *
from ..parallel_3d import *
from ..utils import get_tensor_parallel_mode
from ..vanilla import *
from ._utils import ColossalaiModule

_parallel_linear = {None: VanillaLinear, "1d": Linear1D, "2d": Linear2D, "2.5d": Linear2p5D, "3d": Linear3D}

_parallel_classifier = {
    None: VanillaClassifier,
    "1d": Classifier1D,
    "2d": Classifier2D,
    "2.5d": Classifier2p5D,
    "3d": Classifier3D,
}

_vocab_parallel_classifier = {
    "1d": VocabParallelClassifier1D,
    "2d": VocabParallelClassifier2D,
    "2.5d": VocabParallelClassifier2p5D,
    "3d": VocabParallelClassifier3D,
}


class Linear(ColossalaiModule):
    """Linear layer of colossalai.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    Note: ``kwargs`` would contain different parameters when you use different parallelisms.

    The ``kwargs`` should contain parameters below:
    ::

        Linear1D:
            gather_output: bool (optional, default to be false)
            skip_bias_add: bool (optional, default to be false)
        Linear2D:
            skip_bias_add: bool (optional, default to be false)
        Linear2p5D:
            skip_bias_add: bool (optional, default to be false)
        Linear3D:
            None

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: dtype = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        **kwargs,
    ) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        linear_cls = _parallel_linear[tensor_parallel]
        gather_output = kwargs.pop("gather_output", None)
        if "gather_output" in inspect.signature(linear_cls.__init__).parameters.keys():  # gather_out arg is available
            kwargs["gather_output"] = gather_output
        layer = linear_cls(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            **kwargs,
        )
        super().__init__(layer)


class Classifier(ColossalaiModule):
    """Classifier layer of colossalai.

    Args:
        in_features (int): size of each input sample.
        num_classes (int): number of classes.
        weight (:class:`torch.nn.Parameter`, optional): weight of the classifier, defaults to None.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        weight: nn.Parameter = None,
        bias: bool = True,
        dtype: dtype = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        vocab_parallel_limit: int = 2048,
    ) -> None:
        tensor_parallel = get_tensor_parallel_mode()
        if num_classes <= vocab_parallel_limit or tensor_parallel is None:
            layer = _parallel_classifier[tensor_parallel](
                in_features,
                num_classes,
                weight=weight,
                bias=bias,
                dtype=dtype,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        else:
            layer = _vocab_parallel_classifier[tensor_parallel](
                in_features,
                num_classes,
                weight=weight,
                bias=bias,
                dtype=dtype,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
            )
        super().__init__(layer)
