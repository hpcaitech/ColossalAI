import math

import torch
from torch import Tensor
from torch.nn import Parameter, init as init

from colossalai.context import seed, ParallelMode
from colossalai.core import global_context as gpc
from colossalai.registry import LAYERS
from colossalai.utils import get_current_device
from ._operation import Matmul_AB_2p5D, Add_Bias_2p5D, _LayerNorm_2p5D
from ._utils import get_tesseract_dim_dep_from_env, assert_tesseract_initialization
from .._common_utils import divide, set_tensor_parallel_attribute_by_partition
from ..base_layer import ParallelLayer


@LAYERS.register_module
class Linear2p5D(ParallelLayer):
    """Linear layer for 2.5D parallelism

    :param in_features: size of each input sample
    :type in_features: int
    :param out_features: size of each output sample
    :type out_features: int
    :param bias: If set to ``False``, the layer will not learn an additive bias, defaults to True
    :type bias: bool, optional
    :param dtype: The dtype of parameters, defaults to None
    :type dtype: torch.dtype, optional
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype=None,
                 skip_bias_add: bool = False,
                 init_weight='torch',
                 init_bias='torch'
                 ):
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
        self.hidden_size_per_partition = divide(
            out_features, self.tesseract_dim)

        # create weight, shape: [k/q, h/q]
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}
        self.weight = Parameter(torch.empty(
            self.input_size_per_partition,
            self.hidden_size_per_partition,
            **factory_kwargs))

        # create bias, shape: [h/q]
        if bias:
            self.bias = Parameter(torch.empty(
                self.hidden_size_per_partition,
                **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # initialize parameters
        with seed(ParallelMode.TENSOR):
            self.reset_parameters(init_weight, init_bias)
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        set_tensor_parallel_attribute_by_partition(self.weight, num_partition)
        if self.bias is not None:
            set_tensor_parallel_attribute_by_partition(self.bias, num_partition)

    def reset_parameters(self, init_weight, init_bias) -> None:
        assert init_weight in ('torch', 'jax', 'zero')
        assert init_bias in ('torch', 'jax', 'zero')
        # setting
        fan_in, fan_out = self.in_features, self.out_features

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
        # input: [m/dq, n/q, k/q]
        # output: [m/dq, n/q, h/q]
        out_shape = x.shape[:-1] + (self.hidden_size_per_partition,)

        output = Matmul_AB_2p5D.apply(
            x,
            self.weight,
            self.tesseract_dim,
            out_shape,
            self.row_rank, self.col_rank, self.dep_rank,
            ParallelMode.PARALLEL_2P5D_ROW,
            ParallelMode.PARALLEL_2P5D_COL,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )

        if self.bias is not None:
            if self.skip_bias_add:
                bias = Add_Bias_2p5D.apply(
                    None,
                    self.bias,
                    self.hidden_size_per_partition,
                    self.tesseract_dim,
                    self.row_rank, self.col_rank, self.dep_rank,
                    ParallelMode.PARALLEL_2P5D_COL,
                    True,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size
                )
                return output, bias
            else:
                output = Add_Bias_2p5D.apply(
                    output,
                    self.bias,
                    self.hidden_size_per_partition,
                    self.tesseract_dim,
                    self.row_rank, self.col_rank, self.dep_rank,
                    ParallelMode.PARALLEL_2P5D_COL,
                    False,
                    self.data_parallel_rank,
                    self.pipeline_parallel_rank,
                    self.pipeline_parallel_size,
                    self.tensor_parallel_size
                )
                return output
        else:
            return output


@LAYERS.register_module
class LayerNorm2p5D(ParallelLayer):
    r"""Layer Normalization for 2.5D parallelism

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

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-05,
                 dtype=None
                 ):
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
        self.partitioned_partition = divide(
            normalized_shape, self.tesseract_dim)  # *

        # create parameters
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}

        self.gamma = Parameter(torch.ones(
            self.partitioned_partition,
            **factory_kwargs))
        self.beta = Parameter(torch.zeros(
            self.partitioned_partition,
            **factory_kwargs))

        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        num_partition = gpc.get_world_size(ParallelMode.TENSOR)
        set_tensor_parallel_attribute_by_partition(self.gamma, num_partition)
        set_tensor_parallel_attribute_by_partition(self.beta, num_partition)

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            E_x = torch.sum(x, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(
                E_x, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))
            E_x /= self.normalized_shape

            # Var_x in the block below is the sum of input^2
            Var_x = torch.sum(x * x, dim=-1, keepdim=True)  # [b/q, s, 1]
            torch.distributed.all_reduce(
                Var_x, group=gpc.get_group(ParallelMode.PARALLEL_2P5D_ROW))
            Var_x /= self.normalized_shape

            Var_x = Var_x - E_x * E_x  # variance of x [b/q, s, 1]
            # this time 1/sqrt(Var_x + epsilon)
            Var_x = 1.0 / torch.sqrt(Var_x + self.variance_epsilon)

        output = _LayerNorm_2p5D.apply(x, E_x, Var_x, self.normalized_shape,
                                       ParallelMode.PARALLEL_2P5D_ROW)
        bias = Add_Bias_2p5D.apply(
            None, self.beta, self.partitioned_partition,
            self.tesseract_dim,
            self.row_rank, self.col_rank, self.dep_rank,
            ParallelMode.PARALLEL_2P5D_COL,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size
        )
        scale = Add_Bias_2p5D.apply(
            None, self.gamma, self.partitioned_partition,
            self.tesseract_dim,
            self.row_rank, self.col_rank, self.dep_rank,
            ParallelMode.PARALLEL_2P5D_COL,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size
        )
        output = torch.addcmul(bias, scale, output)
        return output
