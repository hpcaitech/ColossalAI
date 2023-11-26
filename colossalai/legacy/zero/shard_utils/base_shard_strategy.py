from abc import ABC, abstractmethod
from typing import List, Optional

import torch.distributed as dist

from colossalai.legacy.zero.sharded_param.sharded_tensor import ShardedTensor


class BaseShardStrategy(ABC):
    def __init__(self) -> None:
        """Abstract Shard Strategy. Use to shard a tensors on multiple GPUs."""
        super().__init__()

    @abstractmethod
    def shard(self, tensor_list: List[ShardedTensor], process_group: Optional[dist.ProcessGroup] = None):
        pass

    @abstractmethod
    def gather(self, tensor_list: List[ShardedTensor], process_group: Optional[dist.ProcessGroup] = None):
        pass
