import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.utils import get_current_device
from typing import Optional

ZERO_USE_NCCL = False
try:
    import colossal_zero_comm
    ZERO_USE_NCCL = True
except ImportError:
    print("Please pip reinstall Colossalai.")


class ZeroCommWorld(metaclass=SingletonMeta):
    """Zero communicator, used for communications in zero parallel.
    """

    def __init__(self):
        super().__init__()
        self.zero_pg: Optional[ProcessGroup] = None

    @property
    def is_initialized(self):
        return self.zero_pg is not None

    def zero_comm_init(self, comm_group: ProcessGroup):
        if not ZERO_USE_NCCL:
            return

        if self.is_initialized:
            assert self.zero_pg == comm_group, "Cant not initialize zero group twice"
            return

        self.zero_pg = comm_group
        colossal_zero_comm.create_comm(self.zero_pg, get_current_device())

    def zero_all_gather(self, input_tensor: torch.Tensor):
        assert self.zero_pg is not None, "Please initialize zero communication world first"
        rank = dist.get_rank(self.zero_pg)
        world_size = self.zero_pg.size()
        colossal_zero_comm.inplace_all_gather(input_tensor, rank, world_size)


ZeroDist = ZeroCommWorld()
