import math
from typing import Callable, Optional

from colossalai.nn.layer.parallel_1d.layers import Classifier1D
from colossalai.utils import get_current_device
from torch import dtype, nn

from ... import init as init
from ..parallel_1d import *
from ..parallel_2d import *
from ..parallel_2p5d import *
from ..parallel_3d import *
from ..utils import get_tensor_parallel_mode
from ..vanilla import *

_parallel_linear = {'1d': Linear1D, '2d': Linear2D, '2.5d': Linear2p5D, '3d': Linear3D}

_parallel_classifier = {
    'None': VanillaClassifier,
    '1d': Classifier1D,
    '2d': Classifier2D,
    '2.5d': Classifier2p5D,
    '3d': Classifier3D
}


class Linear(nn.Module):
    """
    Linear layer of colossalai

    :param in_features: size of each input sample
    :type in_features: int
    :param out_features: size of each output sample
    :type out_features: int
    :param bias: If set to ``False``, the layer will not learn an additive bias, defaults to True
    :type bias: bool, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param weight_initializer: The intializer of weight, defaults to kaiming uniform initializer
    :type weight_initializer: typing.Callable, optional
    :param bias_initializer: The intializer of bias, defaults to xavier uniform initializer
    :type bias_initializer: typing.Callable, optional
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 **kwargs) -> None:
        super().__init__()
        tensor_parallel = get_tensor_parallel_mode()
        if tensor_parallel == 'None':
            self.layer = nn.Linear(in_features, out_features, bias=bias, device=get_current_device(), dtype=dtype)
            weight_initializer(self.layer.weight, fan_in=in_features, fan_out=out_features)
            if bias:
                bias_initializer(self.layer.bias, fan_in=in_features)
        else:
            self.layer = _parallel_linear[tensor_parallel](
                in_features,
                out_features,
                bias=bias,
                dtype=dtype,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                **kwargs,
            )

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, *args):
        return self.layer(*args)


class Classifier(nn.Module):
    """
    Classifier layer of colossalai

    :param in_features: size of each input sample
    :type in_features: int
    :param num_classes: number of total classes for the dataset
    :type num_classes: int
    :param bias: If set to ``False``, the layer will not learn an additive bias, defaults to True
    :type bias: bool, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param weight_initializer: The intializer of weight, defaults to kaiming uniform initializer
    :type weight_initializer: typing.Callable, optional
    :param bias_initializer: The intializer of bias, defaults to xavier uniform initializer
    :type bias_initializer: typing.Callable, optional
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        weight: nn.Parameter = None,
        bias: bool = True,
        dtype: dtype = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)
    ) -> None:
        super().__init__()
        self.layer = _parallel_classifier[get_tensor_parallel_mode()](
            in_features,
            num_classes,
            weight=weight,
            bias=bias,
            dtype=dtype,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, *args):
        return self.layer(*args)
