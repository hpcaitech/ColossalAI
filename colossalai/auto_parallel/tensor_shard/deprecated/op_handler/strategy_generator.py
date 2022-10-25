from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict
from colossalai.device.device_mesh import DeviceMesh

__all__ = ['IntermediateStrategy', 'StrategyGenerator']


@dataclass
class IntermediateStrategy:
    """
    IntermediateStrategy contains the subset of meta information for ShardingStrategy. It is 
    to store the essential information regarding the tensor sharding and leave other meta information to OperatorHandler.

    Args:
        name (str): name of the sharding strategy.
        dim_partition_dict (Dict[Dict]): stores the tensor to dim partition dict mapping.
        all_reduce_dims (List[int]): stores the dimensions which require an all-reduce operation.
    """
    name: str
    dim_partition_dict: Dict[str, Dict[int, List[int]]]
    all_reduce_axis: List[int] = None


class StrategyGenerator(ABC):
    """
    StrategyGenerator is used to generate the same group of sharding strategies. 
    """

    def __init__(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh

    @abstractmethod
    def generate(self) -> List[IntermediateStrategy]:
        """
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs) -> bool:
        """
        Validate if the operands are of desired shape. 
        If True, means this generator can be used for the current operation.
        """
        pass
