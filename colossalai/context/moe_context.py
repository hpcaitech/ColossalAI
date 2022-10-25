import torch
import torch.distributed as dist

from colossalai.context.parallel_mode import ParallelMode
from colossalai.context.singleton_meta import SingletonMeta
from colossalai.tensor import ProcessGroup

from typing import Tuple


def _check_sanity():
    from colossalai.core import global_context as gpc
    if gpc.tensor_parallel_size > 1 or gpc.pipeline_parallel_size > 1:
        raise NotImplementedError("Moe is not compatible with tensor or "
                                  "pipeline parallel at present.")


class MoeParallelInfo:
    """Moe parallelism information, storing parallel sizes and groups.
    """

    def __init__(self, ep_size: int, dp_size: int):
        _check_sanity()
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.pg = ProcessGroup(tp_degree=ep_size, dp_degree=dp_size)
        self.ep_group = self.pg.tp_process_group()
        self.dp_group = self.pg.dp_process_group()


class MoeContext(metaclass=SingletonMeta):
    """MoE parallel context manager. This class manages different
    parallel groups in MoE context and MoE loss in training.
    """

    def __init__(self):
        self.world_size = 1
        # Users may want to set maximum expert parallel size smaller than the world size
        # since very low bandwidth across nodes may constrain the performance of MoE
        # When we have a maximum expert parallel size, we have a minimum data parallel size naturally
        self.max_ep_size = 1
        self.min_dp_size = 1
        self.aux_loss = None
        self.use_kernel_optim = True

        self.has_setup = False
        self._parallel_info_dict = dict()

    @property
    def parallel_info_dict(self):
        return self._parallel_info_dict

    @property
    def is_initialized(self):
        return self.has_setup

    def setup(self, seed: int, use_kernel_optim: bool = True):
        assert not self.is_initialized, "MoE distributed context shouldn't be set up again"
        _check_sanity()
        assert torch.cuda.is_available(), "MoE requires to enable CUDA first"

        self.world_size = dist.get_world_size()

        from colossalai.core import global_context as gpc
        self.max_ep_size = gpc.config.get('max_ep_size', self.world_size)
        assert self.world_size % self.max_ep_size == 0, \
            "Maximum epxert parallel size must be a factor of the number of GPUs"
        self.min_dp_size = self.world_size // self.max_ep_size

        # Enabling kernel optimization may raise error in some cases
        # Users can close kernel optimization manually
        self.use_kernel_optim = use_kernel_optim

        from .random import moe_set_seed
        moe_set_seed(seed)
        self.has_setup = True

    def get_info(self, num_experts: int) -> Tuple[int, MoeParallelInfo]:
        """Calculate the Data Parallel Group and Expert Parallel Group.

        Parameters
        ----------
        num_experts : int
            The number experts

        Returns
        -------
        int, MoeParallelInfo
            number of local experts, the MoeParallelInfo of the current ep_size
        """

        gt_flag = num_experts % self.max_ep_size == 0    # check whether num_experts is greater
        lt_flag = self.max_ep_size % num_experts == 0    # check whether num_experts is less

        assert gt_flag or lt_flag, "Automatic experts placement dose not not support expert number" \
                                   " is not a multiple of ep size or vice versa."

        # If the number of experts is greater than maximum expert parallel size. a.k.a ep_size,
        # there are multiple experts in each GPU and each GPU has different experts
        # So it's data parallel size is 1
        # Otherwise, there is only one expert in each GPU
        # The data parallel size should be calculated
        dp_size = 1 if gt_flag else self.max_ep_size // num_experts
        ep_size = self.max_ep_size // dp_size

        # Calculate the number of experts for each GPU
        num_local_experts = 1 if lt_flag else num_experts // self.max_ep_size

        # Don't forget to multiply minimum data parallel size
        dp_size *= self.min_dp_size
        if not (ep_size in self.parallel_info_dict):
            self.parallel_info_dict[ep_size] = MoeParallelInfo(ep_size, dp_size)

        return num_local_experts, self.parallel_info_dict[ep_size]

    def set_kernel_not_use(self):
        self.use_kernel_optim = False

    def reset_loss(self):
        self.aux_loss = 0

    def add_loss(self, loss):
        self.aux_loss += loss

    def get_loss(self):
        return self.aux_loss


MOE_CONTEXT = MoeContext()
