import torch

from ..policies.basepolicy import Col_Layer, Layer, Row_Layer
from .shard_config import ShardConfig

dim_mapping = {Col_Layer: 0, Row_Layer: 1}


class Slicer():

    def __init__(
            self,
            shardconfig: ShardConfig    #TODO
    ) -> None:
        self.shardconfig = shardconfig

    def slice_weight_bias(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        policy_layer_cls: Layer,
        n_cast: int = None,
        reversed: bool = False,
    ):
        r"""
        Slice the weight and bias according to policy layer cls
        ``Layer`` -> do nothing
        ``Col_Layer`` -> slice the weight and bias along dim 1
        ``Row_Layer`` -> slice the weight along dim 0 and do not slice bias

        Args:
            weight (:class:`torch.nn.Module`): The weight of the layer
            bias: (:class:`torch.nn.Module`): The bias of the layer
            policy_layer_class (:class:`Policy`): The class represent how to slice the tensor
        """
        if policy_layer_cls == Layer:
            return weight, bias

        dim = dim_mapping[policy_layer_cls] if not reversed else (1 - dim_mapping[policy_layer_cls])
        # print(weight.shape, dim)
        if policy_layer_cls == Col_Layer:
            weight = self.slice_tensor(weight, dim, False, n_cast)
            bias = self.slice_tensor(bias, 0, True, n_cast)
        elif policy_layer_cls == Row_Layer:
            weight = self.slice_tensor(weight, dim, False, n_cast)
        else:
            raise NotImplementedError(f"The policy layer class {policy_layer_cls} is not supported")
        if reversed:
            weight = weight.transpose(0, 1).contiguous()
        return weight, bias

    def slice_tensor(
        self,
        tensor_in: torch.Tensor,
        dim: int,
        is_bias: bool,
        n_cast: int = None,
    ) -> torch.Tensor:
        r"""
        Slice tensor according to the config

        Args:
            tensor_in (:class:`torch.Tensor`): The tensor to slice
            dim (int): The dimension to slice
            is_bias (bool): Whether the tensor is bias
        """
        if tensor_in is None:
            return None
        if not is_bias:
            return self.slice_2d(tensor_in, dim, n_cast)
        else:
            return self.slice_1d(tensor_in, n_cast)

    def slice_2d(
        self,
        tensor: torch.Tensor,
        dim: int,
        n_cast: int = None,
    ) -> torch.Tensor:
        r"""
        Slice the 2D tensor

        Args:
            tensor (:class:`torch.Tensor`): The tensor to slice
            dim (int): The dimension to slice
        """
        assert dim in [0, 1], f"Only support 2D tensor, but got {dim}D tensor"
        if dim == 0:
            return self.slice_row(tensor, n_cast)
        elif dim == 1:
            return self.slice_col(tensor, n_cast)

    def slice_1d(
        self,
        tensor: torch.Tensor,
        n_cast: int = None,
    ) -> torch.Tensor:
        r"""
        Slice the 1D tensor

        Args:
            tensor (:class:`torch.Tensor`): The tensor to slice

        Returns:
            :class:`torch.Tensor`: The sliced tensor
        """
        if n_cast is None:
            return tensor.chunk(self.shardconfig.world_size, dim=0)[self.shardconfig.rank].contiguous()
        else:
            tensor_chunks = tensor.chunk(self.shardconfig.world_size * n_cast, dim=0)
            chunk_list = [
                tensor_chunks[i] for i in range(self.shardconfig.rank, len(tensor_chunks), self.shardconfig.world_size)
            ]
            return torch.cat(chunk_list, dim=0).contiguous()

    def slice_col(
        self,
        tensor: torch.Tensor,
        n_cast: int = None,
    ) -> torch.Tensor:
        r"""
        Slice the tensor in column

        Args:
            tensor (:class:`torch.Tensor`): The tensor to slice

        Returns:
            :class:`torch.Tensor`: The sliced tensor

        """
        if n_cast is None:
            return tensor.chunk(self.shardconfig.world_size, dim=1)[self.shardconfig.rank].contiguous()
        else:
            tensor_chunks = tensor.chunk(self.shardconfig.world_size * n_cast, dim=1)
            chunk_list = [
                tensor_chunks[i] for i in range(self.shardconfig.rank, len(tensor_chunks), self.shardconfig.world_size)
            ]
            return torch.cat(chunk_list, dim=1).contiguous()

    def slice_row(
        self,
        tensor: torch.Tensor,
        n_cast: int = None,
    ) -> torch.Tensor:
        r"""
        Slice the tensor in column

        Args:
            tensor (:class:`torch.Tensor`): The tensor to slice

        Returns:
            :class:`torch.Tensor`: The sliced tensor
        """
        if n_cast is None:
            return tensor.chunk(self.shardconfig.world_size, dim=0)[self.shardconfig.rank].contiguous()
        else:
            tensor_chunks = tensor.chunk(self.shardconfig.world_size * n_cast, dim=0)
            chunk_list = [
                tensor_chunks[i] for i in range(self.shardconfig.rank, len(tensor_chunks), self.shardconfig.world_size)
            ]
            return torch.cat(chunk_list, dim=0).contiguous()
