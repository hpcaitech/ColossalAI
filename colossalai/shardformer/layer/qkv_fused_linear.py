#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

from colossalai.lazy import LazyInitContext
from colossalai.nn import init as init
from colossalai.nn.layer.utils import divide
from colossalai.tensor.d_tensor.api import (
    customized_distributed_tensor_to_existing_param,
    distribute_tensor_with_customization,
    is_customized_distributed_tensor,
    is_distributed_tensor,
    shard_rowwise,
    sharded_tensor_to_existing_param,
)

from ._operation import (
    gather_forward_split_backward,
    linear_with_async_comm,
    matmul_gather_forward_reducescatter_backward,
    matmul_with_async_comm,
    reduce_backward,
    reduce_forward,
    reducescatter_forward_gather_backward,
    split_forward_gather_backward,
)
from .parallel_module import ParallelModule
from .utils import create_randomizer_with_offset

__all__ = ["FusedLinear1D_Col", "FusedLinear1D_Row", "GPT2FusedLinearConv1D_Col", "GPT2FusedLinearConv1D_Row"]

# ====================================
# For GPT Only
# ====================================


def split_fused_qkv_in_gpt2_style(
    qkv: torch.Tensor, n_fused: int, process_group: ProcessGroup, is_transposed: bool = False
):
    """
    The fused qkv tensor looks like [Q1, Q2, K1, K2, V1, V2], this function will split them into [Q1, K1, V1] and [Q2, K2, V2].

    Args:
        qkv (torch.Tensor): The fused qkv tensor.
        n_fused (int): The number items fused together, defaults to 3 (query, key and value).
        process_group (ProcessGroup): The process group for distributed communication.
        is_transposed (bool): generally the tensor is the shape of (out_features, in_features). Set this to True if the tensor is in the shape (in_features, out_features).
    """
    # get the number of slice for the fused qkv
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    order = torch.arange(world_size * n_fused)

    # split the fused qkv
    # from
    # [Q, K, V]
    # to
    # [Q1, Q2, K1, K2, V1, V2]
    if is_transposed:
        weight_chunks = torch.chunk(qkv, world_size * n_fused, dim=-1)
    else:
        weight_chunks = torch.chunk(qkv, world_size * n_fused, dim=0)

    # rearrange the slice into the final order
    # from
    # [Q1, Q2, K1, K2, V1, V2]
    # to
    # [Q1, K1, V1], [Q2, K2, V2]
    weight_chunks_of_current_rank = [weight_chunks[i] for i in order[rank::world_size]]

    if is_transposed:
        weight_of_current_rank = torch.cat(weight_chunks_of_current_rank, dim=-1)
    else:
        weight_of_current_rank = torch.cat(weight_chunks_of_current_rank, dim=0)
    return weight_of_current_rank


def gather_fused_qkv_in_gpt2_style(
    qkv: torch.Tensor, n_fused: int, process_group: ProcessGroup, is_transposed: bool = False
):
    """
    The splitted qkv tensor looks like [Q1, K1, V1] and [Q2, K2, V2], this function will gather them into [Q1, Q2, K1, K2, V1, V2].

    Args:
        qkv (torch.Tensor): The fused qkv tensor.
        n_fused (int): The number items fused together, defaults to 3 (query, key and value).
        process_group (ProcessGroup): The process group for distributed communication.
        is_transposed (bool): generally the tensor is the shape of (out_features, in_features). Set this to True if the tensor is in the shape (in_features, out_features).
    """
    world_size = dist.get_world_size(group=process_group)

    # gather the tensors
    # from
    # [Q1, K1, V1], [Q2, K2, V2]
    # to
    # [Q1, K1, V1, Q2, K2, V2]
    origin_device = qkv.device
    qkv = qkv.cuda()
    gather_list = [torch.zeros_like(qkv) for _ in range(world_size)]
    dist.all_gather(gather_list, qkv, group=process_group)

    if is_transposed:
        gather_weight = torch.cat(gather_list, dim=-1)
    else:
        gather_weight = torch.cat(gather_list, dim=0)
    gather_weight = gather_weight.to(origin_device)
    qkv = qkv.to(origin_device)

    # rearrange the tensor slices
    # from
    # [Q1, K1, V1, Q2, K2, V2]
    # to
    # [Q1, Q2, K1, K2, V1, V2]
    if is_transposed:
        weight_chunks = torch.chunk(gather_weight, world_size * n_fused, dim=-1)
    else:
        weight_chunks = torch.chunk(gather_weight, world_size * n_fused, dim=0)

    reordered_chunk_list = []
    for i in range(n_fused):
        reordered_chunk_list.extend(weight_chunks[i::n_fused])

    if is_transposed:
        reordered_gather_weight = torch.cat(reordered_chunk_list, dim=-1)
    else:
        reordered_gather_weight = torch.cat(reordered_chunk_list, dim=0)
    return reordered_gather_weight


class GPT2FusedLinearConv1D_Col(ParallelModule):
    r"""Linear layer with column parallelism.

    The linear layer is defined as :math:`Y = XA + b`. A is parallelized along
    its second dimension as :math:`A = [A_1, ..., A_p]`. This layer is used to fit `Conv1D` layer (Fused QKV) in gpt2 of huggingface.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (`torch.dtype`): The dtype of parameters, defaults to None.
        device (`torch.device`): The device of parameters, defaults to None.
        n_fused (int): The number items fused, defaults to 3 (QKV).
        process_group (`torch.distributed.ProcessGroup`): The process group to be used for weight sharding and communication, defaults to None.
        seq_parallel_mode (str): If set to ``None``, it will not use sequence parallel, otherwise will use corresponding mode of sequence parallel, defaults to None.
        gather_output (bool, optional): If true, call all-gather on output and make Y available
                    to all GPUs, otherwise, every GPU will have its output
                    which is :math:`Y_i = XA_i`, defaults to False
        skip_bias_add (bool): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to False
        weight_initializer (`typing.Callable`):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (`typing.Callable`):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        process_group: ProcessGroup = None,
        async_communication: bool = False,
        gather_output: bool = False,
        seq_parallel_mode: str = None,
        overlap: bool = False,
        skip_bias_add: bool = False,
        n_fused: int = 3,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        fp8_communication: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.seq_parallel_mode = seq_parallel_mode
        self.overlap = overlap
        self.skip_bias_add = skip_bias_add
        self.device = device
        self.n_fused = n_fused
        self.process_group = process_group
        self.async_communication = async_communication
        self.fp8_communication = fp8_communication

        if skip_bias_add and not bias:
            raise ValueError("cannot skip bias addition if bias is None")

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            # Initialize weight.
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = Parameter(torch.empty(self.in_features, self.out_features, **factory_kwargs))
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight

        def shard_fn(tensor):
            return split_fused_qkv_in_gpt2_style(tensor, self.n_fused, self.process_group, True)

        def gather_fn(tensor):
            return gather_fused_qkv_in_gpt2_style(tensor, self.n_fused, self.process_group, True)

        if not is_customized_distributed_tensor(self.weight):
            with torch.no_grad():
                sharded_weight = distribute_tensor_with_customization(self.weight.data, shard_fn, gather_fn)
            customized_distributed_tensor_to_existing_param(sharded_weight, self.weight)

        if bias:
            if bias_ is None:
                self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
            if not is_customized_distributed_tensor(self.bias):
                with torch.no_grad():
                    sharded_bias = distribute_tensor_with_customization(self.bias.data, shard_fn, gather_fn)
                customized_distributed_tensor_to_existing_param(sharded_bias, self.bias)
        else:
            self.bias = None

        if weight is None:
            # init weights
            self.reset_parameters(weight_initializer, bias_initializer)

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        r"""
        Convert a huggingface layer `Conv1D` in gpt2 to a parallelized linear layer.

        Args:
            module (`nn.Linear`): The module to be converted.
            process_group (`Union[ProcessGroup, List[ProcessGroup]]`): The process group to be used for weight sharding and communication.
            n_fused (int): The number of layers to be fused. In GPT2, Q,K,V are fused in one weight.
        """
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.weight.shape[0]
        out_features = module.weight.shape[1]
        bias = module.bias is not None
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        if out_features < tp_size:
            return module

        if out_features % tp_size != 0:
            raise ValueError(
                f"The size of out_features:{out_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )

        linear_1d = GPT2FusedLinearConv1D_Col(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            *args,
            **kwargs,
        )

        return linear_1d

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with self.randomizer.fork_rng(enable_cpu=True):
            fan_in, fan_out = self.in_features, self.out_features
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        assert (
            input_.shape[-1] == self.weight.shape[0]
        ), "Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.".format(
            input_.shape, self.weight.shape, self.weight.shape[-1]
        )

        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None
        if self.seq_parallel_mode == "split_gather":
            input_parallel = input_
            output_parallel = matmul_gather_forward_reducescatter_backward(
                input_parallel,
                self.weight,
                bias,
                self.process_group,
                True,
                1,
                self.overlap,
                fp8_communication=self.fp8_communication,
            )
        elif self.seq_parallel_mode == "ring":
            input_parallel = input_
            output_parallel = matmul_gather_forward_reducescatter_backward(
                input_parallel,
                self.weight,
                bias,
                self.process_group,
                True,
                1,
                self.overlap,
                True,
                fp8_communication=self.fp8_communication,
            )
        elif self.seq_parallel_mode is None or self.seq_parallel_mode == "ring_attn":
            # Set up backprop all-reduce.
            input_parallel = reduce_backward(input_, self.process_group)
            output_parallel = matmul_with_async_comm(
                input_parallel,
                self.weight,
                bias,
                self.process_group,
                self.async_communication,
                fp8_communication=self.fp8_communication,
            )
        else:
            raise NotImplementedError(f"seq_parallel_mode={self.seq_parallel_mode} is not supported!")

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(
                output_parallel, dim=-1, process_group=self.process_group, fp8_communication=self.fp8_communication
            )
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output


class GPT2FusedLinearConv1D_Row(ParallelModule):
    r"""Linear layer with row parallelism.
    This layer is used to fit `Conv1D` layer (Fused QKV) in gpt2 of huggingface.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (`torch.dtype`): The dtype of parameters, defaults to None.
        parallel_input (bool): If set to ``True``, it's assumed that the input is split, defaults to False.
        skip_bias_add (bool): If set to ``True``, it will skip bias add for linear layer,
        seq_parallel_mode (str): If set to ``None``, it will not use sequence parallel, otherwise will use corresponding mode of sequence parallel, defaults to None.
            which is preserved for kernel fusion, defaults to False
        weight_initializer (:class:`typing.Callable`, optional):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (:class:`typing.Callable`, optional):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        process_group: ProcessGroup = None,
        seq_parallel_mode: str = None,
        parallel_input: bool = True,
        skip_bias_add: bool = False,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        stream_chunk_num: int = 1,
        fp8_communication: bool = False,
    ):
        super().__init__()

        self.stream_chunk_num = stream_chunk_num

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_input = parallel_input
        self.skip_bias_add = skip_bias_add
        self.process_group = process_group
        self.seq_parallel_mode = seq_parallel_mode
        self.num_partitions = dist.get_world_size(self.process_group)
        self.fp8_communication = fp8_communication

        if skip_bias_add and not bias:
            raise ValueError("cannot skip bias addition if bias is None")

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(in_features, self.num_partitions)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            # Initialize weight.
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = Parameter(torch.empty(self.in_features, self.out_features, **factory_kwargs))
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight
        if not is_distributed_tensor(self.weight):
            sharded_weight = shard_rowwise(self.weight.data, self.process_group)
            sharded_tensor_to_existing_param(sharded_weight, self.weight)

        if self.stream_chunk_num > 1:
            # TODO() work for inference only
            self.chunk_weight()
        if bias:
            if bias_ is None:
                self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
        else:
            self.bias = None

        if weight is None:
            # init weights
            self.reset_parameters(weight_initializer, bias_initializer)

    @staticmethod
    def from_native_module(
        module: nn.Linear, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        r"""
        Convert a native PyTorch linear layer to a parallelized linear layer.
        """
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.weight.shape[0]
        out_features = module.weight.shape[1]
        bias = module.bias is not None
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        if in_features < tp_size:
            return module

        if in_features % tp_size != 0:
            raise ValueError(
                f"The size of in_features:{in_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )

        linear_1d = GPT2FusedLinearConv1D_Row(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            *args,
            **kwargs,
        )

        return linear_1d

    def chunk_weight(self):
        self.weight_list = torch.chunk(self.weight, self.stream_chunk_num, dim=0)

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with self.randomizer.fork_rng(enable_cpu=True):
            fan_in, fan_out = self.in_features, self.out_features
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)

            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)
                if self.process_group is None:
                    src_rank = 0
                else:
                    src_rank = dist.distributed_c10d._get_global_rank(self.process_group, 0)

                origin_device = self.bias.device
                self.bias.data = self.bias.cuda()
                dist.broadcast(self.bias, src=src_rank, group=self.process_group)
                self.bias.data = self.bias.to(origin_device)

    def forward(self, input_: Tensor) -> Tensor:
        # Set up backprop all-reduce.
        if self.parallel_input:
            assert (
                input_.shape[-1] == self.weight.shape[0]
            ), "Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.".format(
                input_.shape, self.weight.shape, self.weight.shape[0]
            )
            input_ = input_
        else:
            assert (
                divide(input_.shape[-1], self.num_partitions) == self.weight.shape[0]
            ), "Invalid shapes in Linear1D_Row forward: input={}, weight={}. Expected last dim of input {}.".format(
                input_.shape, self.weight.shape, self.weight.shape[0] * self.num_partitions
            )
            input_ = split_forward_gather_backward(
                input_, dim=-1, process_group=self.process_group, fp8_communication=self.fp8_communication
            )

        if self.stream_chunk_num > 1:
            if self.training:
                raise RuntimeError("use stream_chunk_num=1 in Linear1D_Row for training!")
            with torch.no_grad():
                output_parallel_list = [None for i in range(self.stream_chunk_num)]
                handle_list = []
                for i in range(self.stream_chunk_num):
                    output_parallel_list[i] = torch.matmul(input_, self.weight_list[i])
                    handle = torch.distributed.all_reduce(
                        output_parallel_list[i], group=self.process_group, async_op=True
                    )
                    handle_list.append(handle)
                    # output_parallel_list[i] = reduce_input(output_parallel_list[i], ParallelMode.PARALLEL_1D)
                for handle in handle_list:
                    handle.wait()
                output = torch.cat(output_parallel_list, dim=-1)
        else:
            if self.seq_parallel_mode is None or self.seq_parallel_mode == "ring_attn":
                output_parallel = torch.matmul(input_, self.weight)
                output = reduce_forward(output_parallel, self.process_group, fp8_communication=self.fp8_communication)
            elif self.seq_parallel_mode == "split_gather":
                output_parallel = torch.matmul(input_, self.weight)
                output = reducescatter_forward_gather_backward(
                    output_parallel,
                    self.process_group,
                    1,
                    self.fp8_communication,
                )
            elif self.seq_parallel_mode == "ring":
                output_parallel = torch.matmul(input_, self.weight)
                output = reducescatter_forward_gather_backward(
                    output_parallel,
                    self.process_group,
                    1,
                )
            else:
                raise NotImplementedError(f"seq_parallel_mode={self.seq_parallel_mode} is not supported!")

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias


# ====================================
# For Fused torch.nn.Linear
# ====================================


class FusedLinear1D_Col(ParallelModule):
    r"""Fused Linear layer with column parallelism.

    The linear layer is defined as :math:`Y = XA + b`. A is parallelized along
    its second dimension as :math:`A = [A_1, ..., A_p]`. This layer is used to fit `torch.nn.Linear` layer (Fused QKV) in normal torch layer of huggingface, like SAM.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias, defaults to ``True``.
        dtype (`torch.dtype`): The dtype of parameters, defaults to None.
        device (`torch.device`): The device of parameters, defaults to None.
        n_fused (int): The number items fused, defaults to 3 (QKV).
        process_group (`torch.distributed.ProcessGroup`): The process group to be used for weight sharding and communication, defaults to None.
        gather_output (bool, optional): If true, call all-gather on output and make Y available
                    to all GPUs, otherwise, every GPU will have its output
                    which is :math:`Y_i = XA_i`, defaults to False
        skip_bias_add (bool): If set to ``True``, it will skip bias add for linear layer,
            which is preserved for kernel fusion, defaults to False
        weight_initializer (`typing.Callable`):
            The initializer of weight, defaults to kaiming uniform initializer.
        bias_initializer (`typing.Callable`):
            The initializer of bias, defaults to xavier uniform initializer.

    More details about ``initializer`` please refer to
    `init <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/nn/init.py>`_.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        device: torch.device = None,
        process_group: ProcessGroup = None,
        async_communication: bool = False,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        n_fused: int = 3,
        weight: Optional[Parameter] = None,
        bias_: Optional[Parameter] = None,
        weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
        bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
        fp8_communication: bool = False,
    ):
        super().__init__()
        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.device = device
        self.n_fused = n_fused
        self.process_group = process_group
        self.async_communication = async_communication
        self.fp8_communication = fp8_communication

        if skip_bias_add and not bias:
            raise ValueError("cannot skip bias addition if bias is None")

        # offset the seed with randomizer index and rank
        seed = torch.random.initial_seed()
        self.randomizer = create_randomizer_with_offset(seed, process_group=self.process_group)

        # sanity check
        if weight is not None:
            assert not bias or bias_ is not None, "bias_ must be provided if bias is True when weight is not None"
        else:
            assert bias_ is None, "bias_ must be None if weight is None"

        # Parameters.
        if weight is None:
            # Initialize weight.
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = Parameter(torch.empty(self.out_features, self.in_features, **factory_kwargs))
        else:
            weight.data = weight.data.to(device=device, dtype=dtype)
            self.weight = weight

        def shard_fn(tensor):
            return split_fused_qkv_in_gpt2_style(tensor, self.n_fused, self.process_group, False)

        def gather_fn(tensor):
            return gather_fused_qkv_in_gpt2_style(tensor, self.n_fused, self.process_group, False)

        if not is_customized_distributed_tensor(self.weight):
            with torch.no_grad():
                sharded_weight = distribute_tensor_with_customization(self.weight.data, shard_fn, gather_fn)
            customized_distributed_tensor_to_existing_param(sharded_weight, self.weight)

        if bias:
            if bias_ is None:
                self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
            else:
                bias_.data = bias_.data.to(device=device, dtype=dtype)
                self.bias = bias_
            if not is_customized_distributed_tensor(self.bias):
                with torch.no_grad():
                    sharded_bias = distribute_tensor_with_customization(self.bias.data, shard_fn, gather_fn)
                customized_distributed_tensor_to_existing_param(sharded_bias, self.bias)
        else:
            self.bias = None

        if weight is None:
            # init weights
            self.reset_parameters(weight_initializer, bias_initializer)

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], n_fused: int, *args, **kwargs
    ) -> ParallelModule:
        r"""
        Convert a fused `torch.nn.linear` layer to a parallelized linear layer.

        Args:
            module (`nn.Linear`): The module to be converted.
            process_group (`Union[ProcessGroup, List[ProcessGroup]]`): The process group to be used for weight sharding and communication.
            n_fused (int): The number of layers to be fused. In common, Q,K,V are fused in one weight.
        """
        LazyInitContext.materialize(module)

        # get the attributes
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        linear_1d = FusedLinear1D_Col(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            n_fused=n_fused,
            *args,
            **kwargs,
        )

        # # TODO: copy the sharded weights
        # with torch.no_grad():
        #     sharded_weight = split_fused_qkv_in_gpt2_style(module.weight.data,
        #                                                    n_fused=n_fused,
        #                                                    process_group=process_group,
        #                                                    is_transposed=False)
        #     linear_1d.weight.data.copy_(sharded_weight.data)

        #     if bias:
        #         sharded_bias = split_fused_qkv_in_gpt2_style(module.bias.data,
        #                                                      n_fused=n_fused,
        #                                                      process_group=process_group,
        #                                                      is_transposed=False)
        #         linear_1d.bias.data.copy_(sharded_bias.data)
        return linear_1d

    def reset_parameters(self, weight_initializer, bias_initializer) -> None:
        with self.randomizer.fork_rng(enable_cpu=True):
            fan_in, fan_out = self.in_features, self.out_features
            weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
            if self.bias is not None:
                bias_initializer(self.bias, fan_in=fan_in)

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        assert (
            input_.shape[-1] == self.weight.shape[-1]
        ), "Invalid shapes in Linear1D_Col forward: input={}, weight={}. Expected last dim of input {}.".format(
            input_.shape, self.weight.shape, self.weight.shape[-1]
        )
        # Set up backprop all-reduce.
        # input_parallel = reduce_backward(input_, self.process_group)
        input_parallel = input_

        # Matrix multiply.
        bias = self.bias if not self.skip_bias_add else None

        output_parallel = linear_with_async_comm(input_parallel, self.weight, bias, self.process_group, True)

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_forward_split_backward(
                output_parallel, dim=-1, process_group=self.process_group, fp8_communication=self.fp8_communication
            )
        else:
            output = output_parallel

        if self.skip_bias_add:
            return output, self.bias
        else:
            return output
