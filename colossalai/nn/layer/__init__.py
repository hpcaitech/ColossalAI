from typing import Optional

from colossalai.nn.init import init_bias_, init_weight_
from colossalai.nn.layer.non_parallel_layers.layers import VanillaClassifier
from colossalai.nn.layer.parallel_2d.layers import PatchEmbedding2D
from colossalai.utils import get_current_device
from torch import dtype, nn
from torch.nn.modules.activation import *
from torch.nn.modules.adaptive import *
from torch.nn.modules.batchnorm import *
from torch.nn.modules.channelshuffle import *
from torch.nn.modules.conv import *
from torch.nn.modules.distance import *
from torch.nn.modules.dropout import *
from torch.nn.modules.flatten import *
from torch.nn.modules.fold import *
from torch.nn.modules.instancenorm import *
from torch.nn.modules.linear import *
from torch.nn.modules.normalization import *
from torch.nn.modules.padding import *
from torch.nn.modules.pixelshuffle import *
from torch.nn.modules.pooling import *
from torch.nn.modules.rnn import *
from torch.nn.modules.sparse import *
from torch.nn.modules.transformer import *
from torch.nn.modules.upsampling import *

from .fused_bias_gelu import bias_gelu_impl
from .non_parallel_layers import *
from .parallel_1d import *
from .parallel_2d import *
from .parallel_2p5d import *
from .parallel_3d import *
from .parallel_sequence import *
from .wrapper import *

_parallel_linear = {'1d_col': Linear1D_Col, '1d_row': Linear1D_Row, '2d': Linear2D, '2.5d': Linear2p5D, '3d': Linear3D}

_parallel_classifier = {'2d': Classifier2D, '3d': Classifier3D}

_parallel_layernorm = {'2d': LayerNorm2D, '2p5d': LayerNorm2p5D, '3d': LayerNorm3D}

_parallel_patchembedding = {'2d': PatchEmbedding2D, '3d': PatchEmbedding3D}


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_weight: str = 'torch',
                 init_bias: str = 'torch',
                 tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        if tensor_parallel is None:
            self.layer = nn.Linear(in_features, out_features, bias=bias, device=get_current_device(), dtype=dtype)
            init_weight_(self.layer.weight, in_features, out_features, init_method=init_weight)
            init_bias_(self.layer.bias, in_features, init_method=init_bias)
        else:
            self.layer = _parallel_linear[tensor_parallel](
                in_features,
                out_features,
                bias=bias,
                dtype=dtype,
                init_weight=init_weight,
                init_bias=init_bias,
            )

    def forward(self, *args):
        return self.layer(*args)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-05, dtype=None, tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        if tensor_parallel in [None, '1d']:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps, device=get_current_device(), dtype=dtype)
        else:
            self.norm = _parallel_layernorm[tensor_parallel](normalized_shape, eps=eps, dtype=dtype)

    def forward(self, *args):
        return self.norm(*args)


class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 dtype: dtype = None,
                 flatten: bool = True,
                 init_weight: str = 'torch',
                 init_bias: str = 'torch',
                 tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        if tensor_parallel in [None, '1d']:
            self.embed = VanillaPatchEmbedding(
                img_size,
                patch_size,
                in_chans,
                embed_size,
                dtype=dtype,
                flatten=flatten,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        else:
            self.embed = _parallel_patchembedding[tensor_parallel](
                img_size,
                patch_size,
                in_chans,
                embed_size,
                dtype=dtype,
                flatten=flatten,
                init_weight=init_weight,
                init_bias=init_bias,
            )

    def forward(self, *args):
        return self.embed(*args)


class Classifier(nn.Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: nn.Parameter = None,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_weight: str = 'torch',
                 init_bias: str = 'torch',
                 tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        if tensor_parallel in [None, '1d']:
            self.layer = VanillaClassifier(
                in_features,
                num_classes,
                weight=weight,
                bias=bias,
                dtype=dtype,
                init_weight=init_weight,
                init_bias=init_bias,
            )
        else:
            self.layer = _parallel_classifier[tensor_parallel](
                in_features,
                num_classes,
                weight=weight,
                bias=bias,
                dtype=dtype,
                init_weight=init_weight,
                init_bias=init_bias,
            )

    def forward(self, *args):
        return self.layer(*args)
