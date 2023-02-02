from enum import Enum

__all__ = ['ShardOption']


class ShardOption(Enum):
    """
    This enum class is to define the shard level required in node strategies.

    Notes:
        STANDARD: We do not add any extra shard requirements.
        SHARD: We require the node to be shard using at least one device mesh axis.
        FULL_SHARD: We require the node to be shard using all device mesh axes.
    """
    STANDARD = 0
    SHARD = 1
    FULL_SHARD = 2
