from dataclasses import dataclass

__all__ = ['ShardConfig']


@dataclass
class ShardConfig:
    r"""
    The config for sharding the huggingface model

    Args:
        rank (int): The rank of local process
        world_size (int): The world size of the distributed process
        gather_output (bool): Whether to gather the output of the model of the last layer
    """
    rank: int = None
    world_size: int = None
    gather_output: bool = True
