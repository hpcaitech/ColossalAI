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
from .._common_utils import divide, set_tensor_parallel_attribute
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
                 skip_bias_add: bool = False
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
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()

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
        self.reset_parameters()
        self._set_tensor_parallel_attributes()

    def _set_tensor_parallel_attributes(self):
        set_tensor_parallel_attribute(self.weight)
        if self.bias is not None:
            set_tensor_parallel_attribute(self.bias)

    def reset_parameters(self) -> None:
        # setting
        fan_in = self.in_features
        a = math.sqrt(5)
        nonlinearity = 'leaky_relu'

        # init weight
        std = init.calculate_gain(nonlinearity, a) / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        with seed(ParallelMode.TENSOR):
            init.uniform_(self.weight, -bound, bound)

        # init bias
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            with seed(ParallelMode.TENSOR):
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # input: [m/dq, n/q, k/q]
        # output: [m/dq, n/q, h/q]
        out_shape = x.shape[:-1] + (self.hidden_size_per_partition,)
        output = Matmul_AB_2p5D.apply(
            x,
            self.weight,
            self.tesseract_dim,
            self.tesseract_dep,
            out_shape,
            self.row_rank, self.col_rank, self.dep_rank,
            ParallelMode.PARALLEL_2P5D_ROW,
            ParallelMode.PARALLEL_2P5D_COL,
            ParallelMode.PARALLEL_2P5D_DEP,
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
                    self.tesseract_dim, self.tesseract_dep,
                    self.row_rank, self.col_rank, self.dep_rank,
                    ParallelMode.PARALLEL_2P5D_ROW,
                    ParallelMode.PARALLEL_2P5D_COL,
                    ParallelMode.PARALLEL_2P5D_DEP,
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
                    self.tesseract_dim, self.tesseract_dep,
                    self.row_rank, self.col_rank, self.dep_rank,
                    ParallelMode.PARALLEL_2P5D_ROW,
                    ParallelMode.PARALLEL_2P5D_COL,
                    ParallelMode.PARALLEL_2P5D_DEP,
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
        self.tesseract_dim, self.tesseract_dep = get_tesseract_dim_dep_from_env()

        # partitioning dimension
        self.partitioned_partition = divide(
            normalized_shape, self.tesseract_dim)  # *

        # create parameters
        factory_kwargs = {'device': get_current_device(), 'dtype': dtype}

        if self.row_rank == 0:
            self.gamma = Parameter(torch.ones(
                self.partitioned_partition,
                **factory_kwargs))
            self.beta = Parameter(torch.zeros(
                self.partitioned_partition,
                **factory_kwargs))
        else:
            self.gamma = Parameter(torch.tensor(
                1.0,
                requires_grad=True,
                **factory_kwargs))
            self.beta = Parameter(torch.tensor(
                1.0,
                requires_grad=True,
                **factory_kwargs))
        self._set_tensor_parallel_attribute()

    def _set_tensor_parallel_attribute(self):
        set_tensor_parallel_attribute(self.gamma)
        set_tensor_parallel_attribute(self.beta)

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
                                       ParallelMode.PARALLEL_2P5D_ROW,
                                       ParallelMode.PARALLEL_2P5D_COL,
                                       ParallelMode.PARALLEL_2P5D_DEP)
        bias = Add_Bias_2p5D.apply(
            None, self.beta, self.partitioned_partition,
            self.tesseract_dim, self.tesseract_dep,
            self.row_rank, self.col_rank, self.dep_rank,
            ParallelMode.PARALLEL_2P5D_ROW,
            ParallelMode.PARALLEL_2P5D_COL,
            ParallelMode.PARALLEL_2P5D_DEP,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size
        )
        scale = Add_Bias_2p5D.apply(
            None, self.gamma, self.partitioned_partition,
            self.tesseract_dim, self.tesseract_dep,
            self.row_rank, self.col_rank, self.dep_rank,
            ParallelMode.PARALLEL_2P5D_ROW,
            ParallelMode.PARALLEL_2P5D_COL,
            ParallelMode.PARALLEL_2P5D_DEP,
            True,
            self.data_parallel_rank,
            self.pipeline_parallel_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size
        )
        output = torch.addcmul(bias, scale, output)
        return output
