from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import ParallelModule

from .linear import W8A8B8O8Linear, W8A8BFP32O32LinearSiLU, W8A8BFP32OFP32Linear


def split_row_copy(smooth_linear, para_linear, tp_size=1, tp_rank=0, split_num=1):
    qweights = smooth_linear.weight.split(smooth_linear.out_features // split_num, dim=0)
    if smooth_linear.bias is not None:
        bias = smooth_linear.bias.split(smooth_linear.out_features // split_num, dim=0)

    smooth_split_out_features = para_linear.out_features // split_num

    for i in range(split_num):
        para_linear.weight[i * smooth_split_out_features : (i + 1) * smooth_split_out_features, :] = qweights[i][
            tp_rank * smooth_split_out_features : (tp_rank + 1) * smooth_split_out_features, :
        ]

        if para_linear.bias is not None:
            para_linear.bias[:, i * smooth_split_out_features : (i + 1) * smooth_split_out_features] = bias[i][
                :, tp_rank * smooth_split_out_features : (tp_rank + 1) * smooth_split_out_features
            ]


def split_column_copy(smooth_linear, para_linear, tp_rank=0, split_num=1):
    qweights = smooth_linear.weight.split(smooth_linear.in_features // split_num, dim=-1)

    smooth_split_in_features = para_linear.in_features // split_num

    for i in range(split_num):
        para_linear.weight[:, i * smooth_split_in_features : (i + 1) * smooth_split_in_features] = qweights[i][
            :, tp_rank * smooth_split_in_features : (tp_rank + 1) * smooth_split_in_features
        ]

    if smooth_linear.bias is not None:
        para_linear.bias.copy_(smooth_linear.bias)


class RowW8A8B8O8Linear(W8A8B8O8Linear, ParallelModule):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__(in_features, out_features, alpha, beta)
        self.process_group = None
        self.tp_size = 1
        self.tp_rank = 0

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        out_features = module.out_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if out_features < tp_size:
            return module

        if out_features % tp_size != 0:
            raise ValueError(
                f"The size of out_features:{out_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = RowW8A8B8O8Linear(module.in_features, module.out_features // tp_size)
        linear_1d.tp_size = tp_size
        linear_1d.tp_rank = tp_rank
        linear_1d.process_group = process_group
        linear_1d.a = module.a.clone().detach()
        linear_1d.b = module.b.clone().detach()
        split_row_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        return linear_1d


class ColW8A8B8O8Linear(W8A8B8O8Linear, ParallelModule):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__(in_features, out_features, alpha, beta)
        self.process_group = None
        self.tp_size = 1
        self.tp_rank = 0

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.in_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if in_features < tp_size:
            return module

        if in_features % tp_size != 0:
            raise ValueError(
                f"The size of in_features:{in_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = ColW8A8B8O8Linear(module.in_features // tp_size, module.out_features)
        linear_1d.tp_size = tp_size
        linear_1d.tp_rank = tp_rank
        linear_1d.process_group = process_group
        linear_1d.a = torch.tensor(module.a)
        linear_1d.b = torch.tensor(module.b)

        split_column_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        if linear_1d.bias is not None:
            linear_1d.bias = linear_1d.bias // tp_size

        return linear_1d

    @torch.no_grad()
    def forward(self, x):
        output = super().forward(x)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output


class RowW8A8BFP32O32LinearSiLU(W8A8BFP32O32LinearSiLU, ParallelModule):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__(in_features, out_features, alpha, beta)
        self.process_group = None
        self.tp_size = 1
        self.tp_rank = 0

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        out_features = module.out_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if out_features < tp_size:
            return module

        if out_features % tp_size != 0:
            raise ValueError(
                f"The size of out_features:{out_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = RowW8A8BFP32O32LinearSiLU(module.in_features, module.out_features // tp_size)
        linear_1d.tp_size = tp_size
        linear_1d.tp_rank = tp_rank
        linear_1d.process_group = process_group
        linear_1d.a = module.a.clone().detach()

        split_row_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        return linear_1d


class RowW8A8BFP32OFP32Linear(W8A8BFP32OFP32Linear, ParallelModule):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__(in_features, out_features, alpha, beta)
        self.process_group = None
        self.tp_size = 1
        self.tp_rank = 0

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        out_features = module.out_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if out_features < tp_size:
            return module

        if out_features % tp_size != 0:
            raise ValueError(
                f"The size of out_features:{out_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = RowW8A8BFP32OFP32Linear(module.in_features, module.out_features // tp_size)
        linear_1d.tp_size = tp_size
        linear_1d.tp_rank = tp_rank
        linear_1d.process_group = process_group
        linear_1d.a = module.a.clone().detach()

        split_row_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        return linear_1d


class ColW8A8BFP32OFP32Linear(W8A8BFP32OFP32Linear, ParallelModule):
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__(in_features, out_features, alpha, beta)
        self.process_group = None
        self.tp_size = 1
        self.tp_rank = 0

    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        # get the attributes
        in_features = module.in_features

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        tp_rank = dist.get_rank(process_group)

        if in_features < tp_size:
            return module

        if in_features % tp_size != 0:
            raise ValueError(
                f"The size of in_features:{in_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )
        linear_1d = ColW8A8BFP32OFP32Linear(module.in_features // tp_size, module.out_features)
        linear_1d.tp_size = tp_size
        linear_1d.tp_rank = tp_rank
        linear_1d.process_group = process_group
        linear_1d.a = module.a.clone().detach()

        split_column_copy(module, linear_1d, tp_rank=tp_rank, **kwargs)
        if linear_1d.bias is not None:
            linear_1d.bias = linear_1d.bias / tp_size

        return linear_1d

    @torch.no_grad()
    def forward(self, x):
        output = super().forward(x)
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.process_group)
        return output
