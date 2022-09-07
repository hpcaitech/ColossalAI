import math
from collections import OrderedDict
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.communication import broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.global_variables import tensor_parallel_env as env
from colossalai.nn import init as init
from colossalai.registry import LAYERS
from colossalai.utils.checkpointing import (broadcast_state_dict, gather_tensor_parallel_state_dict,
                                            partition_tensor_parallel_state_dict)
from colossalai.utils.cuda import get_current_device
from torch import Tensor
from torch.nn import Parameter

from ..base_layer import ParallelLayer
from ..utils import divide, set_tensor_parallel_attribute_by_partition, to_2tuple
from ._operation import (Matmul_AB_2p5D, Matmul_ABT_2p5D, add_bias_2p5d, all_gather_tensor_2p5d, classifier_2p5d,
                         layernorm_2p5d, reduce_scatter_tensor_2p5d, split_batch_2p5d)
from ._utils import assert_tesseract_initialization, get_tesseract_dim_dep_from_env


@LAYERS.register_module
class Linear2p5D(ParallelLayer):
    r"""Linear layer for 2.5D parallelism.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        skip_bias_add (bool, optional): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to False.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 skip_bias_add: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        # parallel setting
        assert_tesseract_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
        self.dep_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)
        self.tesseract_dim, _ = get_tesseract_dim_dep_from_env()

        # partitioning dimension
        self.input_size_per_partition = divide(in_features, self.tesseract_dim)
        self.hidden_size_per_partition = divide(out_features, self.tesseract_dim)

        # create weight, shape: [k/q, h/q]
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(
            torch.empty(self.input_size_per_partition, self.hidden_size_per_partition, **factory_kwargs))

        # create bias, shape: [h/q]
        if bias:
            self.bias = Parameter(torch.empty(self.hidden_size_per_partition, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # initialize parameters
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim**2)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.tesseract_dim)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight.transpose(0, 1)
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        # broadcast in dep groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0 and \
                gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW) == 0:
            broadcast_state_dict(local_state, ParallelMode.PARALLEL_2P5D_DEP)
        # partition in column groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_COL,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
            )
        # partition in row groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_ROW,
            dims={
                weight_key: -1,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: True
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP) == 0:
            weight_key = prefix + 'weight'
            bias_key = prefix + 'bias'
            local_state = OrderedDict({weight_key: self.weight})
            if self.bias is not None:
                local_state[bias_key] = self.bias

            # gather in row groups
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
                keep_vars=keep_vars,
            )
            # gather in column groups
            if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW) == 0:
                local_state = gather_tensor_parallel_state_dict(
                    local_state,
                    ParallelMode.PARALLEL_2P5D_COL,
                    dims={
                        weight_key: 0,
                        bias_key: 0
                    },
                    partition_states={
                        weight_key: True,
                        bias_key: False
                    },
                    keep_vars=keep_vars,
                )
            if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
                local_state[weight_key] = local_state[weight_key].transpose(0, 1)
                destination.update(local_state)

    def forward(self, x: Tensor) -> Tensor:
        # input: [m/dq, n/q, k/q]
        # output: [m/dq, n/q, h/q]
        out_shape = x.shape[:-1] + (self.hidden_size_per_partition,)

        output = Matmul_AB_2p5D.apply(
            x,
            self.weight,
            self.tesseract_dim,
            out_shape,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            ParallelMode.PARALLEL_2P5D_ROW,
            ParallelMode.PARALLEL_2P5D_COL,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2p5d(None, self.bias, self.hidden_size_per_partition, self.tesseract_dim, self.row_rank,
                                     self.col_rank, self.dep_rank, ParallelMode.PARALLEL_2P5D_COL, True,
                                     self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                     self.tensor_parallel_size)
                return output, bias
            else:
                output = add_bias_2p5d(output, self.bias, self.hidden_size_per_partition, self.tesseract_dim,
                                       self.row_rank, self.col_rank, self.dep_rank, ParallelMode.PARALLEL_2P5D_COL,
                                       False, self.data_parallel_rank, self.pipeline_parallel_rank,
                                       self.pipeline_parallel_size, self.tensor_parallel_size)
                return output
        else:
            return output


@LAYERS.register_module
class LayerNorm2p5D(ParallelLayer):
    r"""Layer Normalization for 2.5D parallelism.

    Args:
        normalized_shape (int): input shape from an expected input of size.
            :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
            \times \ldots \times \text{normalized_shape}[-1]]`
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (float, optional): a value added to the denominator for numerical stability, defaults to 1e-05.
        bias (bool, optional): Whether to add a bias, defaults to ``True``.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-05, bias=True, dtype=None):
        super().__init__()

        # layer norm config
        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        # parallel setting
        assert_tesseract_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
        self.dep_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)
        self.tesseract_dim, _ = get_tesseract_dim_dep_from_env()

        # partitioning dimension
        self.partitioned_partition = divide(normalized_shape, self.tesseract_dim)    # *

        # create parameters
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}

        self.weight = Parameter(torch.ones(self.partitioned_partition, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(self.partitioned_partition, **factory_kwargs))
        else:
            self.bias = None

        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.tesseract_dim)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            bias = state_dict.pop(bias_key, None)
            if bias is not None:
                local_state[bias_key] = bias

        # partition in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
            )
        # partition in column groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: True
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict({weight_key: self.weight})
        if self.bias is not None:
            local_state[bias_key] = self.bias

        # gather in column groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: True
            },
            keep_vars=keep_vars,
        )
        # gather in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: 0,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            E_x = torch.sum(x, dim=-1, keepdim=True)    # [b/q, s, 1]
            torch.distributed.all_reduce(E_x, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))
            E_x /= self.normalized_shape

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(x * x, dim=-1, keepdim=True)    # [b/q, s, 1]
            torch.distributed.all_reduce(Var_x, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x    # variance of x [b/q, s, 1]
            # this time 1/sqrt(Var_x + epsilon)
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)

        output = layernorm_2p5d(x, E_x, Var_x, self.normalized_shape, ParallelMode.PARALLEL_2P5D_ROW)
        scale = add_bias_2p5d(None, self.weight, self.partitioned_partition, self.tesseract_dim, self.row_rank,
                              self.col_rank, self.dep_rank, ParallelMode.PARALLEL_2P5D_COL, True,
                              self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                              self.tensor_parallel_size)
        if self.bias is not None:
            bias = add_bias_2p5d(None, self.bias, self.partitioned_partition, self.tesseract_dim, self.row_rank,
                                 self.col_rank, self.dep_rank, ParallelMode.PARALLEL_2P5D_COL, True,
                                 self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                 self.tensor_parallel_size)
            output = torch.addcmul(bias, scale, output)
        else:
            output = torch.mul(scale, output)
        return output


@LAYERS.register_module
class PatchEmbedding2p5D(ParallelLayer):
    r"""2D Image to Patch Embedding.

    Args:
        img_size (int): image size.
        patch_size (int): patch size.
        in_chans (int): number of channels of input image.
        embed_size (int): size of embedding.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        flatten (bool, optional): whether to flatten output tensor, defaults to True.
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.
        position_embed_initializer (:class:`typing.Callable`, optional):
            The initializer of position embedding, defaults to zeros initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
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

        assert_tesseract_initialization()
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_size = embed_size
        self.embed_size_per_partition = embed_size // self.tesseract_dim**2

        with seed(ParallelMode.TENSOR):
            self.weight = Parameter(
                torch.empty((self.embed_size_per_partition, in_chans, *self.patch_size),
                            device=get_current_device(),
                            dtype=dtype))
            self.bias = Parameter(torch.empty(self.embed_size_per_partition, device=get_current_device(), dtype=dtype))

            self.cls_token = Parameter(
                torch.zeros((1, 1, self.embed_size_per_partition), device=get_current_device(), dtype=dtype))
            self.pos_embed = Parameter(
                torch.zeros((1, self.num_patches + 1, self.embed_size_per_partition),
                            device=get_current_device(),
                            dtype=dtype))

        self.reset_parameters(weight_initializer, bias_initializer, position_embed_initializer)
        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim**2)
        set_tensor_parallel_attribute_by_partition(self.bias, self.tesseract_dim**2)
        set_tensor_parallel_attribute_by_partition(self.cls_token, self.tesseract_dim**2)
        set_tensor_parallel_attribute_by_partition(self.pos_embed, self.tesseract_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer):
        with seed(ParallelMode.TENSOR):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_out = self.embed_size
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            bias_initializer(self.bias, fan_in=fan_in)
            position_embed_initializer(self.pos_embed)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        cls_token_key = prefix + 'cls_token'
        pos_embed_key = prefix + 'pos_embed'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight
            # bias
            bias = state_dict.pop(bias_key, None)
            if bias is not None:
                local_state[bias_key] = bias
            # cls token
            cls_token = state_dict.pop(cls_token_key, None)
            if cls_token is not None:
                local_state[cls_token_key] = cls_token
            # pos embed
            pos_embed = state_dict.pop(pos_embed_key, None)
            if pos_embed is not None:
                local_state[pos_embed_key] = pos_embed

        # partition in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: 0,
                    bias_key: 0,
                    cls_token_key: -1,
                    pos_embed_key: -1
                },
                partition_states={
                    weight_key: True,
                    bias_key: True,
                    cls_token_key: True,
                    pos_embed_key: True
                },
            )
        # partition in column groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: 0,
                bias_key: 0,
                cls_token_key: -1,
                pos_embed_key: -1
            },
            partition_states={
                weight_key: True,
                bias_key: True,
                cls_token_key: True,
                pos_embed_key: True
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        cls_token_key = prefix + 'cls_token'
        pos_embed_key = prefix + 'pos_embed'
        local_state = OrderedDict({
            weight_key: self.weight,
            bias_key: self.bias,
            cls_token_key: self.cls_token,
            pos_embed_key: self.pos_embed
        })

        # gather in column groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: 0,
                bias_key: 0,
                cls_token_key: -1,
                pos_embed_key: -1
            },
            partition_states={
                weight_key: True,
                bias_key: True,
                cls_token_key: True,
                pos_embed_key: True
            },
            keep_vars=keep_vars,
        )
        # gather in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: 0,
                    bias_key: 0,
                    cls_token_key: -1,
                    pos_embed_key: -1
                },
                partition_states={
                    weight_key: True,
                    bias_key: True,
                    cls_token_key: True,
                    pos_embed_key: True
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_2p5d(input_, 0)

        B, C, H, W = input_.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        weight = all_gather_tensor_2p5d(self.weight, 0, ParallelMode.PARALLEL_2P5D_COL)
        bias = all_gather_tensor_2p5d(self.bias, 0, ParallelMode.PARALLEL_2P5D_COL)

        output = F.conv2d(input_, weight, bias, stride=self.patch_size)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)    # BCHW -> BNC

        cls_token = all_gather_tensor_2p5d(self.cls_token, -1, ParallelMode.PARALLEL_2P5D_COL)
        pos_embed = all_gather_tensor_2p5d(self.pos_embed, -1, ParallelMode.PARALLEL_2P5D_COL)
        cls_token = cls_token.expand(output.shape[0], -1, -1)
        output = torch.cat((cls_token, output), dim=1)
        output = output + pos_embed

        return output


@LAYERS.register_module
class Embedding2p5D(ParallelLayer):
    r"""Embedding for 2.5D parallelism.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:``torch.nn.functional.embedding`` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about initializer please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()

        assert_tesseract_initialization()
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim // self.tesseract_dim**2

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim**2)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight

        # partition in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={weight_key: -1},
                partition_states={weight_key: True},
            )
        # partition in column groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={weight_key: -1},
            partition_states={weight_key: True},
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        local_state = OrderedDict({weight_key: self.weight})

        # gather in column groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={weight_key: -1},
            partition_states={weight_key: True},
            keep_vars=keep_vars,
        )
        # gather in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={weight_key: -1},
                partition_states={weight_key: True},
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        input_ = split_batch_2p5d(input_, 0)

        weight = all_gather_tensor_2p5d(self.weight, -1, ParallelMode.PARALLEL_2P5D_COL)

        output = F.embedding(input_, weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        return output


@LAYERS.register_module
class VocabParallelEmbedding2p5D(ParallelLayer):
    """Embedding parallelized in the vocabulary dimension.

    Args:
        num_embeddings (int): number of embeddings.
        embedding_dim (int): dimension of embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx do not contribute to the gradient;
            therefore, the embedding vector at padding_idx is not updated during training,
            i.e. it remains as a fixed “pad”, defaults to None.
        dtype (:class:`torch.dtype`, optional): The dtype of parameters, defaults to None.
        weight_initializer (:class:`typing.Callable`, optional):
            he initializer of weight, defaults to normal initializer.

    The ``args`` and ``kwargs`` used in :class:``torch.nn.functional.embedding`` should contain:
    ::

        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is
                    renormalized to have norm max_norm. Note: this will modify weight in-place.
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse
                    of frequency of the words in the mini-batch. Default False.
        sparse (bool, optional): If True, gradient w.r.t. weight will be a sparse tensor. Default False.

    More details about ``args`` and ``kwargs`` could be found in
    `Embedding <https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html#torch.nn.functional.embedding>`_.

    More details about initializer please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        assert_tesseract_initialization()
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()
        self.num_embeddings_per_partition = divide(self.num_embeddings, self.tesseract_dim)
        self.embed_dim_per_partition = divide(self.embed_dim, self.tesseract_dim)
        tensor_parallel_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
        self.vocab_start_index = tensor_parallel_rank * self.num_embeddings_per_partition
        self.vocab_end_index = self.vocab_start_index + self.num_embeddings_per_partition

        self.weight = Parameter(
            torch.empty((self.num_embeddings_per_partition, self.embed_dim_per_partition),
                        device=get_current_device(),
                        dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim**2)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None and \
                self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
            with torch.no_grad():
                self.weight[self.padding_idx - self.vocab_start_index].fill_(0)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            weight = state_dict.pop(weight_key, None)
            if weight is not None:
                local_state[weight_key] = weight

        # partition in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={weight_key: -1},
                partition_states={weight_key: True},
            )
        # partition in column groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={weight_key: 0},
            partition_states={weight_key: True},
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        local_state = OrderedDict({weight_key: self.weight})

        # gather in column groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={weight_key: 0},
            partition_states={weight_key: True},
            keep_vars=keep_vars,
        )
        # gather in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={weight_key: -1},
                partition_states={weight_key: True},
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, *self.embed_args,
                                      **self.embed_kwargs)

        # Mask the output embedding.
        output_parallel[input_mask, :] = 0.
        # Reduce across all the model parallel GPUs.
        output = reduce_scatter_tensor_2p5d(output_parallel, 0, ParallelMode.PARALLEL_2P5D_COL)
        return output


@LAYERS.register_module
class Classifier2p5D(ParallelLayer):
    r"""Classifier for 2.5D parallelism.

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

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        assert_tesseract_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
        self.dep_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()

        # partitioning dimension
        self.input_size_per_partition = divide(self.in_features, self.tesseract_dim**2)

        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.num_classes, self.input_size_per_partition, device=get_current_device(), dtype=dtype))
            self.has_weight = True
        if bias:
            self.bias = Parameter(torch.zeros(self.num_classes, device=get_current_device(), dtype=dtype))
        else:
            self.bias = None

        self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes
            col_src_rank = gpc.get_ranks_in_group(ParallelMode.PARALLEL_2P5D_COL)[0]
            row_src_rank = gpc.get_ranks_in_group(ParallelMode.PARALLEL_2P5D_ROW)[0]

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, col_src_rank, ParallelMode.PARALLEL_2P5D_COL)
                broadcast(self.bias, row_src_rank, ParallelMode.PARALLEL_2P5D_ROW)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        # partition in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
            )
        # partition in column groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: -1,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: False
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def _save_to_global_state_dict(self, destination, prefix, keep_vars):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        local_state = OrderedDict()
        if self.has_weight:
            local_state[weight_key] = self.weight
        if self.bias is not None:
            local_state[bias_key] = self.bias

        # gather in column groups
        local_state = gather_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: -1,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: False
            },
            keep_vars=keep_vars,
        )
        # gather in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = gather_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: False
                },
                keep_vars=keep_vars,
            )
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            destination.update(local_state)

    def forward(self, input_: Tensor) -> Tensor:
        out_shape = input_.shape[:-1] + (self.num_classes,)

        return classifier_2p5d(input_, self.weight, self.bias, self.tesseract_dim, out_shape, self.row_rank,
                               self.col_rank, ParallelMode.PARALLEL_2P5D_ROW, ParallelMode.PARALLEL_2P5D_COL,
                               self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                               self.tensor_parallel_size)


@LAYERS.register_module
class VocabParallelClassifier2p5D(ParallelLayer):
    r"""Vocab parallel classifier layer for 2.5D parallelism.

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

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        # parallel setting
        assert_tesseract_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)
        self.dep_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)
        self.tesseract_dim, _ = get_tesseract_dim_dep_from_env()

        # partitioning dimension
        self.input_size_per_partition = divide(in_features, self.tesseract_dim)
        self.hidden_size_per_partition = divide(num_classes, self.tesseract_dim)

        # create weight, shape: [k/q, h/q]
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        if weight is not None:
            self.weight = weight
            self.has_weight = False
        else:
            self.weight = Parameter(
                torch.empty(self.hidden_size_per_partition, self.input_size_per_partition, **factory_kwargs))
            self.has_weight = True
        # create bias, shape: [h/q]
        if bias:
            self.bias = Parameter(torch.empty(self.hidden_size_per_partition, **factory_kwargs))
        else:
            self.bias = None

        # initialize parameters
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()
        env.vocab_parallel = True

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.tesseract_dim**2)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.tesseract_dim)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.num_classes
        if self.has_weight:
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def _load_from_global_state_dict(self, state_dict, prefix, *args, **kwargs):
        local_state = OrderedDict()
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        if gpc.get_local_rank(ParallelMode.TENSOR) == 0:
            # weight
            if self.has_weight:
                weight = state_dict.pop(weight_key, None)
                if weight is not None:
                    local_state[weight_key] = weight
            # bias
            if self.bias is not None:
                bias = state_dict.pop(bias_key, None)
                if bias is not None:
                    local_state[bias_key] = bias

        # partition in row groups
        if gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL) == 0:
            local_state = partition_tensor_parallel_state_dict(
                local_state,
                ParallelMode.PARALLEL_2P5D_ROW,
                dims={
                    weight_key: -1,
                    bias_key: 0
                },
                partition_states={
                    weight_key: True,
                    bias_key: True
                },
            )
        # partition in column groups
        local_state = partition_tensor_parallel_state_dict(
            local_state,
            ParallelMode.PARALLEL_2P5D_COL,
            dims={
                weight_key: 0,
                bias_key: 0
            },
            partition_states={
                weight_key: True,
                bias_key: True
            },
        )

        super()._load_from_global_state_dict(local_state, prefix, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        # input: [m/dq, n/q, k/q]
        # output: [m/dq, n/q, h/q]
        out_shape = x.shape[:-1] + (self.hidden_size_per_partition,)

        output = Matmul_ABT_2p5D.apply(
            x,
            self.weight,
            self.tesseract_dim,
            out_shape,
            self.row_rank,
            self.col_rank,
            self.dep_rank,
            ParallelMode.PARALLEL_2P5D_ROW,
            ParallelMode.PARALLEL_2P5D_COL,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )

        if self.bias is not None:
            output = add_bias_2p5d(output, self.bias, self.hidden_size_per_partition, self.tesseract_dim, self.row_rank,
                                   self.col_rank, self.dep_rank, ParallelMode.PARALLEL_2P5D_COL, False,
                                   self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                   self.tensor_parallel_size)
        return output
