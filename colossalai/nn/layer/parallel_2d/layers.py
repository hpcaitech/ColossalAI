import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.communication import broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.nn import init as init
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from torch import Tensor, dtype
from torch.nn import Parameter

from ..utils import divide, set_tensor_parallel_attribute_by_partition, to_2tuple
from ..base_layer import ParallelLayer
from ._operation import Matmul_AB_2D, add_bias_2d, all_gather_weight_2d, classifier_2d, layernorm_2d
from ._utils import assert_summa_initialization, get_summa_dim_from_env


@LAYERS.register_module
class Linear2D(ParallelLayer):
    """
    Linear layer for 2D parallelism

    :param in_features: size of each input sample
    :type in_features: int
    :param out_features: size of each output sample
    :type out_features: int
    :param bias: If set to ``False``, the layer will not learn an additive bias, defaults to True
    :type bias: bool, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param skip_bias_add: If set to ``True``, it will skip bias add for linear layer, which is preserved for kernel fusion, defaults to False
    :type skip_bias_add: bool, optional
    :param weight_initializer: The intializer of weight, defaults to kaiming uniform initializer
    :type weight_initializer: typing.Callable, optional
    :param bias_initializer: The intializer of bias, defaults to xavier uniform initializer
    :type bias_initializer: typing.Callable, optional
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype=None,
                 skip_bias_add: bool = False,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add

        # parallel settings
        assert_summa_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.input_size_per_partition = divide(self.in_features, self.summa_dim)
        self.hidden_size_per_partition = divide(self.out_features, self.summa_dim)

        # create weight, shape: [k/q, h/q]
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(
            torch.empty(self.input_size_per_partition, self.hidden_size_per_partition, **factory_kwargs))

        # create bias, shape: [h/q]
        if bias:
            self.bias = Parameter(torch.empty(divide(self.out_features, self.summa_dim**2), **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # initialize parameters
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(weight_initializer, bias_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        fan_in, fan_out = self.in_features, self.out_features
        weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
        if self.bias is not None:
            bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, x: Tensor) -> Tensor:
        # input: [m/q, n/q, k/q]
        # output: [m/q, n/q, h/q]
        out_shape = x.shape[:-1] + (self.hidden_size_per_partition, )

        output = Matmul_AB_2D.apply(x, self.weight, self.summa_dim, out_shape, self.row_rank, self.col_rank,
                                    ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL, self.data_parallel_rank,
                                    self.pipeline_parallel_rank, self.pipeline_parallel_size, self.tensor_parallel_size)

        if self.bias is not None:
            if self.skip_bias_add:
                bias = add_bias_2d.apply(None, self.bias, self.hidden_size_per_partition, self.row_rank, self.col_rank,
                                         ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL, True,
                                         self.data_parallel_rank, self.pipeline_parallel_rank,
                                         self.pipeline_parallel_size, self.tensor_parallel_size)
                return output, bias
            else:
                output = add_bias_2d.apply(output, self.bias, self.hidden_size_per_partition, self.row_rank,
                                           self.col_rank, ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL,
                                           False, self.data_parallel_rank, self.pipeline_parallel_rank,
                                           self.pipeline_parallel_size, self.tensor_parallel_size)
                return output
        else:
            return output


@LAYERS.register_module
class LayerNorm2D(ParallelLayer):
    r"""
    Layer Normalization for 2D parallelism

    :param normalized_shape: input shape from an expected input
        of size. :math:`[* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1] \times \ldots \times \text{normalized_shape}[-1]]`
        If a single integer is used, it is treated as a singleton list, and this module will
        normalize over the last dimension which is expected to be of that specific size.
    :type normalized_shape: int
    :param eps: a value added to the denominator for numerical stability, defaults to 1e-05
    :type eps: float, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-05, dtype=None):
        super().__init__()

        # layer norm config
        self.normalized_shape = normalized_shape
        self.variance_epsilon = eps

        # parallel setting
        assert_summa_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.partitioned_partition = divide(normalized_shape, self.summa_dim**2)

        # create parameters
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}

        self.gamma = Parameter(torch.ones(self.partitioned_partition, **factory_kwargs))
        self.beta = Parameter(torch.zeros(self.partitioned_partition, **factory_kwargs))

        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.gamma, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.beta, self.summa_dim**2)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            E_x = torch.sum(x, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(E_x, group=gpc.get_group(ParallelMode.PARALLEL_2D_ROW))
            E_x /= self.normalized_shape

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(x * x, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(Var_x, group=gpc.get_group(ParallelMode.PARALLEL_2D_ROW))
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x  # variance of x [b/q, s, 1]
            # this time 1/sqrt(Var_x + epsilon)
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)

        output = layernorm_2d.apply(x, E_x, Var_x, self.normalized_shape, ParallelMode.PARALLEL_2D_ROW,
                                    ParallelMode.PARALLEL_2D_COL)
        bias = add_bias_2d.apply(None, self.beta, self.partitioned_partition, self.row_rank, self.col_rank,
                                 ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL, True,
                                 self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                 self.tensor_parallel_size)
        scale = add_bias_2d.apply(None, self.gamma, self.partitioned_partition, self.row_rank, self.col_rank,
                                  ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL, True,
                                  self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                  self.tensor_parallel_size)
        output = torch.addcmul(bias, scale, output)
        return output


@LAYERS.register_module
class PatchEmbedding2D(ParallelLayer):
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
                 dtype: dtype = None,
                 flatten: bool = True,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
                 position_embed_initializer: Callable = init.zeros_()):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.embed_size = embed_size
        self.embed_size_per_partition = embed_size // (self.summa_dim**2)

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
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.cls_token, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.pos_embed, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer):
        with seed(ParallelMode.TENSOR):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            fan_out = self.embed_size
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            bias_initializer(self.bias, fan_in=fan_in)
            position_embed_initializer(self.pos_embed)

    def forward(self, input_: Tensor) -> Tensor:
        B, C, H, W = input_.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        weight = all_gather_weight_2d.apply(self.weight, 0, self.summa_dim, ParallelMode.PARALLEL_2D_COL)
        bias = all_gather_weight_2d.apply(self.bias, 0, self.summa_dim, ParallelMode.PARALLEL_2D_COL)

        output = F.conv2d(input_, weight, bias, stride=self.patch_size)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)  # BCHW -> BNC

        cls_token = all_gather_weight_2d.apply(self.cls_token, -1, self.summa_dim, ParallelMode.PARALLEL_2D_COL)
        pos_embed = all_gather_weight_2d.apply(self.pos_embed, -1, self.summa_dim, ParallelMode.PARALLEL_2D_COL)
        cls_token = cls_token.expand(output.shape[0], -1, -1)
        output = torch.cat((cls_token, output), dim=1)
        output = output + pos_embed

        return output


@LAYERS.register_module
class Embedding2D(ParallelLayer):
    """
    Embedding for 2D parallelism

    :param num_embeddings: number of embeddings
    :type num_embeddings: int
    :param embedding_dim: dimension of embedding
    :type embedding_dim: int
    :param padding_idx: index of padding, defaults to None
    :type padding_idx: int, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    :param weight_initializer: The intializer of weight, defaults to normal initializer
    :type weight_initializer: typing.Callable, optional
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int = None,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.normal_(),
                 *args,
                 **kwargs):
        super().__init__()

        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = divide(embedding_dim, self.summa_dim**2)

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = Parameter(
            torch.empty((num_embeddings, embed_dim_per_partition), device=get_current_device(), dtype=dtype))

        self.reset_parameters(weight_initializer)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)

    def reset_parameters(self, weight_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.num_embeddings, self.embed_dim
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input_: Tensor) -> Tensor:
        weight = all_gather_weight_2d.apply(self.weight, -1, self.summa_dim, ParallelMode.PARALLEL_2D_COL)

        output = F.embedding(input_, weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        return output


@LAYERS.register_module
class Classifier2D(ParallelLayer):
    """
    Classifier for 2D parallelism

    :param in_features: size of each input sample
    :type in_features: int
    :param num_classes: number of classes
    :type num_classes: int
    :param weight: weight of the classifier, defaults to True
    :type weight: torch.nn.Parameter, optional
    :param bias: If set to ``False``, the layer will not learn an additive bias, defaults to ``True``
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
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: dtype = None,
                 weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
                 bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1)):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        assert_summa_initialization()
        self.row_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)
        self.col_rank = gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)
        self.summa_dim = get_summa_dim_from_env()

        # partitioning dimension
        self.input_size_per_partition = divide(self.in_features, self.summa_dim**2)

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
            set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes
            col_src_rank = gpc.get_ranks_in_group(ParallelMode.PARALLEL_2D_COL)[0]
            row_src_rank = gpc.get_ranks_in_group(ParallelMode.PARALLEL_2D_ROW)[0]

            if self.has_weight:
                weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                broadcast(self.bias, col_src_rank, ParallelMode.PARALLEL_2D_COL)
                broadcast(self.bias, row_src_rank, ParallelMode.PARALLEL_2D_ROW)

    def forward(self, input_: Tensor) -> Tensor:
        out_shape = input_.shape[:-1] + (self.num_classes, )

        return classifier_2d.apply(input_, self.weight, self.bias, self.summa_dim, out_shape, self.row_rank,
                                   self.col_rank, ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL,
                                   self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                   self.tensor_parallel_size)
