from dataclasses import dataclass


@dataclass
class ShardConfig:
    """
    The config for sharding the huggingface model for test
    """
    fp16: bool
    num_gpus: int
    rank: int
    backend="nccl"
    verbose: str = 'simple'
    seed: int = None
    require_grad: bool = False
    master_addr: str = "127.0.0.1"
    master_port: int = 29500