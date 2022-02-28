import math
from typing import Callable

import torch
import torch.nn.functional as F
from colossalai.context import seed
from colossalai.nn import init as init
from colossalai.registry import LAYERS
from colossalai.utils.cuda import get_current_device
from torch import Tensor
from torch import nn as nn

from ..utils import to_2tuple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class WrappedDropout(nn.Module):
    """Same as torch.nn.Dropout. But it is wrapped with the context of seed manager.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False, mode=None):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        if mode is None:
            self.func = self.nonefunc
        else:
            self.func = self.normalfunc
            self.mode = mode

    def nonefunc(self, inputs):
        return F.dropout(inputs, self.p, self.training, self.inplace)

    def normalfunc(self, inputs):
        with seed(self.mode):
            return F.dropout(inputs, self.p, self.training, self.inplace)

    def forward(self, inputs):
        return self.func(inputs)


class WrappedDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Here, it is wrapped with the context of seed manager.
    """

    def __init__(self, p: float = 0., mode=None):
        super().__init__()
        self.p = p
        self.mode = mode
        if self.mode is None:
            self.func = self.nonefunc
        else:
            self.func = self.normalfunc
            self.mode = mode

    def nonefunc(self, inputs):
        return drop_path(inputs, self.p, self.training)

    def normalfunc(self, inputs):
        with seed(self.mode):
            return drop_path(inputs, self.p, self.training)

    def forward(self, inputs):
        return self.func(inputs)


@LAYERS.register_module
class VanillaPatchEmbedding(nn.Module):
    """ 
    2D Image to Patch Embedding

    :param img_size: image size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param in_chans: number of channels of input image
    :type in_chans: int
    :param embed_size: size of embedding
    :type embed_size: int
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param flatten: whether to flatten output tensor, defaults to True
    :type flatten: bool, optional
    :param weight_initializer: The intializer of weight, defaults to kaiming uniform initializer
    :type weight_initializer: typing.Callable, optional
    :param bias_initializer: The intializer of bias, defaults to xavier uniform initializer
    :type bias_initializer: typing.Callable, optional
    :param position_embed_initializer: The intializer of position embedding, defaults to zero
    :type position_embed_initializer: typing.Callable, optional
    """

    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 flatten: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 position_embed_initializer: Callable = init.zeros_()):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.weight = nn.Parameter(
            torch.empty((embed_size, in_chans, *self.patch_size), device=get_current_device(), dtype=dtype))
        self.bias = nn.Parameter(torch.empty(embed_size, device=get_current_device(), dtype=dtype))
        self.cls_token = nn.Parameter(torch.zeros((1, 1, embed_size), device=get_current_device(), dtype=dtype))
        self.pos_embed = nn.Parameter(
            torch.zeros((1, self.num_patches + 1, embed_size), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer, bias_initializer, position_embed_initializer)

    def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        bias_initializer(self.bias, fan_in=fan_in)
        position_embed_initializer(self.pos_embed)

    def forward(self, input_: Tensor) -> Tensor:
        B, C, H, W = input_.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        output = F.conv2d(input_, self.weight, self.bias, stride=self.patch_size)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)  # BCHW -> BNC

        cls_token = self.cls_token.expand(output.shape[0], -1, -1)
        output = torch.cat((cls_token, output), dim=1)
        output = output + self.pos_embed
        return output


@LAYERS.register_module
class VanillaClassifier(nn.Module):
    """
    Dense linear classifier

    :param in_features: size of each input sample
    :type in_features: int
    :param num_classes: number of classes
    :type num_classes: int
    :param weight: weight of the classifier, defaults to True
    :type weight: torch.nn.Parameter, optional
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
                 num_classes: int,
                 weight: nn.Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = nn.Parameter(
                torch.empty(self.num_classes, self.in_features, device=get_current_device(), dtype=dtype))
            self.has_weight = True
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_classes, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)

    def reset_parameters(self, weight_initializer, bias_initializer):
        fan_in, fan_out = self.in_features, self.num_classes

        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, self.weight, self.bias)
