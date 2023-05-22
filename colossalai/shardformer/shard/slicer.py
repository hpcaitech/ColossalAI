import os
from typing import Dict, Tuple
from dataclasses import dataclass

import torch
import torch.distributed as dist
from ..policies.basepolicy import Layer, Col_Layer, Row_Layer
from .shardconfig import ShardConfig


dim_mapping = {Col_Layer: 1, Row_Layer: 0}

class Slicer():

    def __init__(
        self, 
        shardconfig: ShardConfig #TODO
    ) -> None:
        self.shardconfig = shardconfig

    
    def slice_weight_bias(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        policy_layer_cls: Layer,
    ):
        """
        Slice the weight and bias according to policy layer cls
        Layer -> do nothing
        Col_Layer -> slice the weight and bias along dim 1
        Row_Layer -> slice the weight along dim 0 and do not slice bias

        Args:
            weight: The weight of the layer
            bias: The bias of the layer
            policy_layer_class: The class represent how to slice the tensor
        """
        if policy_layer_cls == Layer:
            return weight, bias
        elif policy_layer_cls == Col_Layer:
            weight = self.slice_tensor(weight, 1, False)
            bias = self.slice_tensor(bias, 0, True)
        elif policy_layer_cls == Row_Layer:
            weight = self.slice_tensor(weight, 0, False)
        else:
            raise NotImplementedError(f"The policy layer class {policy_layer_cls} is not supported")
        return weight, bias
    

    def slice_weight(
        self,
        weight: torch.Tensor,
        policy_layer_cls: Layer,
    ) -> torch.Tensor:
        """
        Slice the weight and bias according to the shardconfig

        Args:
            weight: The weight of the layer
            bias: The bias of the layer
            policy_layer_class: The class represent how to slice the tensor
        """
        if weight is not None:
            dim = dim_mapping[policy_layer_cls]
            weight = self.slice_tensor(weight, dim, False)
        return weight


    def slice_bias(
        self,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Slice the bias according to the shardconfig
        
        Args:
            bias: The bias of the layer
        """
        assert bias is not None, "The bias is None"
        if bias is not None:
            bias = self.slice_tensor(bias, 1, True)
        return bias


    def slice_tensor(
        self,
        tensor_in: torch.Tensor,
        dim: int,
        is_bias: bool,
    ) -> torch.Tensor:
        """
        Slice tensor according to the config
        """
        if tensor_in is None:
            return None
        if not is_bias:
            return self.slice_2d(tensor_in, dim)
        else:
            return self.slice_1d(tensor_in)


    def slice_2d(
        self,
        tensor: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """
        Slice the 2D tensor 

        Args:
            tensor: The tensor to slice
        """
        assert dim in [0,1], f"Only support 2D tensor, but got {dim}D tensor"
        if dim == 0:
            return self.slice_row(tensor)
        elif dim == 1:
            return self.slice_col(tensor)


    def slice_1d(
        self,
        tensor: torch.Tensor,
        dim: int = None,
    ) -> torch.Tensor:
        """
        Slice the 1D tensor 

        Args:
            tensor: The tensor to slice
        """
        delta = (tensor.shape[0] + self.shardconfig.world_size - 1) // self.shardconfig.world_size
        down_idx = self.shardconfig.rank * delta
        up_idx = down_idx + delta
        return tensor[down_idx:up_idx]

    def slice_col(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Slice the tensor in column

        Args:
            tensor: The tensor to slice
        """
        delta = (tensor.shape[0] + self.shardconfig.world_size - 1) // self.shardconfig.world_size
        down_idx = self.shardconfig.rank * delta
        up_idx = down_idx + delta
        return tensor[down_idx:up_idx,:]


    def slice_row(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Slice the tensor in column

        Args:
            tensor: The tensor to slice
        """
        delta = (tensor.shape[1] + self.shardconfig.world_size - 1) // self.shardconfig.world_size
        down_idx = self.shardconfig.rank * delta
        up_idx = down_idx + delta
        return tensor[:,down_idx:up_idx]
    