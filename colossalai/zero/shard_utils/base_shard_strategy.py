from abc import ABC, abstractmethod
from colossalai.zero.sharded_param.sharded_tensor import ShardedTensor
from torch import Tensor
import torch.distributed as dist
from typing import List, Optional


class BaseShardStrategy(ABC):
    """
    BaseShardStrategy defines the interfaces to standardize the process of sharding and
    gather. Any class which inherits this class should ensure that gather is the reverse
    operation of shard.

    Example:
        # need to make sure that 
        # gather_tensor_list and tensor_list are equal
        tensor_list = [a, b, c, d]
        strategy = BaseShardStrategy()
        sharded_tensor_list = strategy.shard(tensor_list)
        gathered_tensor_list = strategy.shard(sharded_tensor_list)
    """

    def __init__(self, process_group: Optional[dist.ProcessGroup] = None) -> None:
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.local_rank = dist.get_rank(self.process_group)
        super().__init__()

    @abstractmethod
    def shard(self, tensor_list: List[Tensor]) -> List[Tensor]:
        r"""
        sharded the memory of tensor on multiple processes.
        """
        pass

    @abstractmethod
    def gather(self, tensor_list: List[Tensor]) -> List[Tensor]:
        r"""
        duplicate tensor payload on each processes.
        """
        pass
