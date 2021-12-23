import math
from typing import Callable, Optional

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

from .. import init as init

from .vanilla import *
from .parallel_1d import *
from .parallel_2d import *
from .parallel_2p5d import *
from .parallel_3d import *
from .parallel_sequence import *

_parallel_linear = {'1d_col': Linear1D_Col, '1d_row': Linear1D_Row, '2d': Linear2D, '2.5d': Linear2p5D, '3d': Linear3D}

_parallel_classifier = {
    None: VanillaClassifier,
    '1d': VanillaClassifier,
    '2d': Classifier2D,
    '2.5d': Classifier2p5D,
    '3d': Classifier3D
}

_parallel_layernorm = {'2d': LayerNorm2D, '2.5d': LayerNorm2p5D, '3d': LayerNorm3D}

_parallel_embedding = {'3d': Embedding3D}

_parallel_patchembedding = {
    None: VanillaPatchEmbedding,
    '1d': VanillaPatchEmbedding,
    '2d': PatchEmbedding2D,
    '2.5d': PatchEmbedding2p5D,
    '3d': PatchEmbedding3D
}


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 tensor_parallel: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__()
        if tensor_parallel is None:
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


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps=1e-05, dtype=None, tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        if tensor_parallel in [None, '1d']:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps, device=get_current_device(), dtype=dtype)
        else:
            self.norm = _parallel_layernorm[tensor_parallel](normalized_shape, eps=eps, dtype=dtype)

    @property
    def weight(self):
        return self.norm.weight

    @property
    def bias(self):
        return self.norm.bias

    def forward(self, *args):
        return self.norm(*args)


class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 tensor_parallel: Optional[str] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        if tensor_parallel in [None, '1d']:
            self.embed = nn.Embedding(num_embeddings,
                                      embedding_dim,
                                      padding_idx=padding_idx,
                                      device=get_current_device(),
                                      dtype=dtype,
                                      *args,
                                      **kwargs)
            weight_initializer(self.embed.weight, fan_in=num_embeddings, fan_out=embedding_dim)
        else:
            self.embed = _parallel_embedding[tensor_parallel](
                num_embeddings,
                embedding_dim,
                padding_idx=padding_idx,
                dtype=dtype,
                weight_initializer=weight_initializer,
                *args,
                **kwargs,
            )

    @property
    def weight(self):
        return self.embed.weight

    def forward(self, *args):
        return self.embed(*args)


class PatchEmbedding(nn.Module):
    def __init__(self,
                 img_size: int,
                 patch_size: int,
                 in_chans: int,
                 embed_size: int,
                 dtype: dtype = None,
                 flatten: bool = True,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 position_embed_initializer: Callable = init.zeros_(),
                 tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        self.embed = _parallel_patchembedding[tensor_parallel](
            img_size,
            patch_size,
            in_chans,
            embed_size,
            dtype=dtype,
            flatten=flatten,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            position_embed_initializer=position_embed_initializer,
        )

    @property
    def weight(self):
        return self.embed.weight

    @property
    def bias(self):
        return self.embed.bias

    @property
    def pos_embed(self):
        return self.embed.pos_embed

    @property
    def cls_token(self):
        return self.embed.cls_token

    def forward(self, *args):
        return self.embed(*args)


class Classifier(nn.Module):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: nn.Parameter = None,
                 bias: bool = True,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 tensor_parallel: Optional[str] = None) -> None:
        super().__init__()
        self.layer = _parallel_classifier[tensor_parallel](
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
