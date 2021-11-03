import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.communication import all_reduce, broadcast
from colossalai.context import ParallelMode, seed
from colossalai.core import global_context as gpc
from colossalai.nn.init import init_bias_, init_weight_
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from torch import Tensor, dtype
from torch.nn import Parameter
from torch.nn import init as init

from .._common_utils import (divide, set_tensor_parallel_attribute_by_partition, to_2tuple)
from ..base_layer import ParallelLayer
from ._operation import (Matmul_AB_2D, Matmul_ABT_2D, add_bias_2d, all_gather_weight_2d, layernorm_2d, split_batch_2d,
                         classifier_2d)
from ._utils import assert_summa_initialization, get_summa_dim_from_env


@LAYERS.register_module
class Linear2D(ParallelLayer):
    """ Linear layer for 2D parallelism

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
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype=None,
                 skip_bias_add: bool = False,
                 init_weight='torch',
                 init_bias='torch'):
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
            self.reset_parameters(init_weight, init_bias)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)

    def reset_parameters(self, init_weight, init_bias) -> None:
        assert init_weight in ('torch', 'jax', 'zero')
        assert init_bias in ('torch', 'jax', 'zero')
        # setting
        fan_in, fan_out = self.in_features, self.out_features

        with seed(ParallelMode.TENSOR):
            # init weight
            if init_weight == 'torch':
                a = math.sqrt(5)
                nonlinearity = 'leaky_relu'
                std = init.calculate_gain(nonlinearity, a) / math.sqrt(fan_in)
                bound = math.sqrt(3.0) * std
                init.uniform_(self.weight, -bound, bound)
            elif init_weight == 'jax':
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                a = math.sqrt(3.0) * std
                init.uniform_(self.weight, -a, a)
            elif init_weight == 'zero':
                init.zeros_(self.weight)

            # init bias
            if self.bias is not None:
                if init_bias == 'torch':
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(self.bias, -bound, bound)
                elif init_bias == 'jax':
                    init.normal_(self.bias, std=1e-6)
                elif init_bias == 'zero':
                    init.zeros_(self.bias)

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
    r"""Layer Normalization for 2D parallelism

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
    """ 2D Image to Patch Embedding

    :param img_size: iamge size
    :type img_size: int
    :param patch_size: patch size
    :type patch_size: int
    :param embed_dim: dimension of embedding
    :type embed_dim: int
    :param in_chans: number of channels of input image, defaults to 3
    :type in_chans: int, optional
    :param flatten: whether to flatten output tensor, defaults to True
    :type flatten: bool, optional
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

        assert_summa_initialization()
        self.summa_dim = get_summa_dim_from_env()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
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

        self.reset_parameters(init_weight, init_bias)
        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)
        set_tensor_parallel_attribute_by_partition(self.bias, self.summa_dim**2)

    def reset_parameters(self, init_weight, init_bias):
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
            fan_out *= self.summa_dim
            init_weight_(self.weight, fan_in, fan_out, init_method=init_weight)
            init_bias_(self.bias, fan_in, init_method=init_bias)
            init_pos_embed = None if init_weight == 'torch' else init_weight
            init_bias_(self.pos_embed, fan_in, init_method=init_pos_embed)

    def forward(self, input_: Tensor) -> Tensor:
        B, C, H, W = input_.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        input_ = split_batch_2d(input_)

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
class Classifier2D(ParallelLayer):
    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 weight: Parameter = None,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_weight: str = 'torch',
                 init_bias: str = 'torch'):
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

        self.reset_parameters(init_weight, init_bias)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        if self.has_weight:
            set_tensor_parallel_attribute_by_partition(self.weight, self.summa_dim**2)

    def reset_parameters(self, init_weight, init_bias) -> None:
        with seed(ParallelMode.TENSOR):
            fan_in, fan_out = self.in_features, self.num_classes
            col_src_rank = gpc.get_ranks_in_group(ParallelMode.PARALLEL_2D_COL)[0]
            row_src_rank = gpc.get_ranks_in_group(ParallelMode.PARALLEL_2D_ROW)[0]

            if self.has_weight:
                init_weight_(self.weight, fan_in, fan_out, init_method=init_weight)

            if self.bias is not None:
                init_bias_(self.bias, fan_in, init_method=init_bias)
                broadcast(self.bias, col_src_rank, ParallelMode.PARALLEL_2D_COL)
                broadcast(self.bias, row_src_rank, ParallelMode.PARALLEL_2D_ROW)

    def forward(self, input_: Tensor) -> Tensor:
        # input: [m/q, n/q, k/q]
        # output: [m/q, n/q, h/q]
        out_shape = input_.shape[:-1] + (self.num_classes, )

        # output = Matmul_ABT_2D.apply(input_, self.weight, self.summa_dim, out_shape, self.row_rank, self.col_rank,
        #                             ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL, self.data_parallel_rank,
        #                             self.pipeline_parallel_rank, self.pipeline_parallel_size, self.tensor_parallel_size)

        # if self.bias is not None:
        #     if self.skip_bias_add:
        #         bias = add_bias_2d.apply(None, self.bias, self.num_classes, self.row_rank, self.col_rank,
        #                                  ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL, True,
        #                                  self.data_parallel_rank, self.pipeline_parallel_rank,
        #                                  self.pipeline_parallel_size, self.tensor_parallel_size)
        #         return output, bias
        #     else:
        #         output = add_bias_2d.apply(output, self.bias, self.num_classes, self.row_rank,
        #                                    self.col_rank, ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL,
        #                                    False, self.data_parallel_rank, self.pipeline_parallel_rank,
        #                                    self.pipeline_parallel_size, self.tensor_parallel_size)
        #         return output
        # else:
        #     return output
        return classifier_2d.apply(input_, self.weight, self.bias, self.summa_dim, out_shape, self.row_rank,
                                   self.col_rank, ParallelMode.PARALLEL_2D_ROW, ParallelMode.PARALLEL_2D_COL,
                                   self.data_parallel_rank, self.pipeline_parallel_rank, self.pipeline_parallel_size,
                                   self.tensor_parallel_size)
