import torch.nn.functional as F
import torch
from torch import nn as nn
from torch import dtype, Tensor
from colossalai.registry import LAYERS
from .._common_utils import to_2tuple
from colossalai.utils import get_current_device
from colossalai.nn.init import init_weight_, init_bias_


@LAYERS.register_module
class VanillaPatchEmbedding(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 dtype: dtype = None,
                 flatten: bool = True,
                 init_weight: str = 'torch',
                 init_bias: str = 'torch'):
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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_size))

        self.reset_parameters(init_weight, init_bias)

    def reset_parameters(self, init_weight, init_bias):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        init_weight_(self.weight, fan_in, fan_out, init_method=init_weight)
        init_bias_(self.bias, fan_in, init_method=init_bias)
        init_pos_embed = None if init_weight == 'torch' else init_weight
        init_bias_(self.pos_embed, fan_in, init_method=init_pos_embed)

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
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: nn.Parameter = None,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_weight: str = 'torch',
                 init_bias: str = 'torch'):
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

        self.reset_parameters(init_weight, init_bias)

    def reset_parameters(self, init_weight, init_bias) -> None:
        fan_in, fan_out = self.in_features, self.num_classes

        if self.has_weight:
            init_weight_(self.weight, fan_in, fan_out, init_method=init_weight)

        if self.bias is not None:
            init_bias_(self.bias, fan_in, init_method=init_bias)

    def forward(self, input_: Tensor) -> Tensor:
        return F.linear(input_, self.weight, self.bias)
