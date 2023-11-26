from dataclasses import dataclass
from enum import Enum

__all__ = ["SolverOptions", "SolverPerference", "DataloaderOption", "ShardOption"]


class SolverPerference(Enum):
    """
    This enum class is to define the solver preference.
    """

    STANDARD = 0
    DP = 1
    TP = 2


class ShardOption(Enum):
    """
    This enum class is to define the shard level required in node strategies.

    Notes:
        STANDARD: We do not add any extra shard requirements.
        SHARD: We require the node to be shard using at least one device mesh axis.
        SHARD_ONE_AXIS: We require the node to be shard using the last device mesh axis.
        FULL_SHARD: We require the node to be shard using all device mesh axes.
        TP_SHARD: We require the node to be shard using tensor parallel strategies on last device mesh axis.
        TP_FULL_SHARD: We require the node to be shard using tensor parallel strategies on all device mesh axes.
    """

    STANDARD = 0
    SHARD = 1
    SHARD_LAST_AXIS = 2
    FULL_SHARD = 3


class DataloaderOption(Enum):
    """
    This enum class is to define the dataloader option.
    """

    REPLICATED = 0
    DISTRIBUTED = 1


@dataclass
class SolverOptions:
    """
    SolverOptions is a dataclass used to configure the preferences for the parallel execution plan search.
    """

    solver_perference: SolverPerference = SolverPerference.STANDARD
    dataloader_option: DataloaderOption = DataloaderOption.REPLICATED
    shard_option: ShardOption = ShardOption.STANDARD
