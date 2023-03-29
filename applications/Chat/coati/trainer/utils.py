import torch.distributed as dist


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
