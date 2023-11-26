from typing import Dict, List

from colossalai.auto_parallel.tensor_shard.sharding_strategy import (
    MemoryCost,
    OperationData,
    ShardingStrategy,
    TrainCycleItem,
)
from colossalai.device.device_mesh import DeviceMesh

from .strategy_generator import StrategyGenerator

__all__ = ["PlaceholderGenerator"]


class PlaceholderGenerator(StrategyGenerator):
    """
    PlaceholderGenerator is a generic class to generate strategies for placeholder node.
    """

    def __init__(
        self, operation_data_mapping: Dict[str, OperationData], device_mesh: DeviceMesh, placeholder_option: str
    ):
        super().__init__(operation_data_mapping, device_mesh)
        self.placeholder_option = placeholder_option

    def validate(self) -> bool:
        return super().validate()

    def update_compute_cost(self, strategy: ShardingStrategy):
        compute_cost = TrainCycleItem(fwd=10, bwd=10, total=20)
        strategy.compute_cost = compute_cost

    def update_memory_cost(self, strategy: ShardingStrategy):
        """
        Compute the memory cost per device with this specific strategy.
        """
        forward_size_mapping = {"output": self._compute_size_in_bytes(strategy, "output")}

        # compute fwd cost incurred
        # fwd_cost = output
        fwd_activation_cost = sum([v for k, v in forward_size_mapping.items()])
        fwd_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=0)

        bwd_mem_cost = MemoryCost(activation=0, parameter=0)

        # compute total cost
        total_mem_cost = MemoryCost(activation=fwd_activation_cost, parameter=0)
        memory_cost = TrainCycleItem(fwd=fwd_mem_cost, bwd=bwd_mem_cost, total=total_mem_cost)
        strategy.memory_cost = memory_cost

    def replica_placeholder(self) -> ShardingStrategy:
        """
        Generate replica strategy for placeholder node.
        """
        dim_partition_dict_mapping = {
            "output": {},
        }
        communication_action_mapping = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = "Replica Placeholder"

        strategy = self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

        return strategy

    def distributed_placeholder(self, mesh_list) -> ShardingStrategy:
        """
        Generate distributed strategy for placeholder node.
        """
        dim_partition_dict_mapping = {
            "output": {0: mesh_list},
        }
        communication_action_mapping = {}
        sharding_spec_mapping = self.to_sharding_spec_mapping(dim_partition_dict_mapping)

        name = "Distributed Placeholder"

        strategy = self.get_sharding_strategy(
            name=name,
            sharding_spec_mapping=sharding_spec_mapping,
            communication_action_mapping=communication_action_mapping,
        )

        return strategy

    def collate_strategies(self) -> List[ShardingStrategy]:
        strategy_list = []
        if self.placeholder_option == "distributed":
            mesh_list = [0, 1]
            distributed_strategy = self.distributed_placeholder(mesh_list)
            strategy_list.append(distributed_strategy)
        else:
            assert (
                self.placeholder_option == "replicated"
            ), f"placeholder_option {self.placeholder_option} is not supported"
            replicated_strategy = self.replica_placeholder()
            strategy_list.append(replicated_strategy)

        return strategy_list
