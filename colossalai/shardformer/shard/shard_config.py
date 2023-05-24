from dataclasses import dataclass

__all__ = ['ShardConfig']


@dataclass
class ShardConfig:
    """
    The config for sharding the huggingface model for test
    """
    rank: int
    fp16: bool = True
    num_gpus: int = 2
    world_size: int = 2
    backend = "nccl"
    verbose: str = 'simple'
    seed: int = None
    require_grad: bool = False
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
