from typing import Tuple

import torch
import torch.distributed as dist

from colossalai.context.singleton_meta import SingletonMeta
from colossalai.tensor.moe_tensor.api import get_moe_info
from colossalai.tensor.moe_tensor.moe_info import MoeParallelInfo


class MoEManager(metaclass=SingletonMeta):
    """MoE manager. This class manages different
    parallel groups in MoE context and MoE loss in training.
    """

    def __init__(self):
        self.parallel = None
        self.mode = None
        self.use_ep_inside = None
        self.world_size = None
        self._parallel_info_dict = dict()

        # router
        self.router_aux_loss = []
        self.router_z_loss = []

        # fixed mode
        self.pp_size = None
        self.dp_size = None
        self.ep_size = None

        # dynamic mode
        # Users may want to set maximum expert parallel size smaller than the world size
        # since very low bandwidth across nodes may constrain the performance of MoE
        # When we have a maximum expert parallel size, we have a minimum data parallel size naturally
        self.max_ep_size = None

        self.has_setup = False

    @property
    def parallel_info_dict(self):
        return self._parallel_info_dict

    @property
    def is_initialized(self):
        return self.has_setup

    def setup(
        self,
        parallel: str = None,
        mode: str = "dynamic",
        max_ep_size: int = 8,
        fixed_dp_size: int = 0,
        fixed_ep_size: int = 0,
        fixed_pp_size: int = 0,
        use_ep_inside: bool = True,
    ) -> None:
        """
        Setup MoE distributed context.

        Args:
            seed (int): Random seed. Defaults to 42.
            use_kernel_optim (bool, optional): Use cuda kernel. Defaults to True.
            parallel (bool, optional): Parallel mode, should be EP, TP or None. Defaults to None.
            mode (str, optional): Should be "fixed" or "dynamic". Defaults to "dynamic".
                In fixed mode, the ep size and dp size is fixed.
                In dynamic mode, the ep size and dp size will be changed according to num experts.
            max_ep_size (int, optional): Max ep size in dynamic mode. Defaults to 8.
            fixed_dp_size (int, optional): Fixed dp size in fixed mode. Defaults to 0.
            fixed_ep_size (int, optional): Fixed ep size in fixed mode. Defaults to 0.
            fixed_pp_size (int, optional): Fixed pp size in fixed mode. Defaults to 0.
            use_ep_inside (bool, optional): Use ep inside dp if True, dp inside ep if False. Defaults to True.
        """
        assert not self.is_initialized, "MoE distributed context shouldn't be set up again"
        assert torch.cuda.is_available(), "MoE requires to enable CUDA first"

        self.parallel = parallel
        self.use_ep_inside = use_ep_inside
        self.world_size = dist.get_world_size()

        # init by mode
        self.mode = mode
        assert self.mode in ["fixed", "dynamic"], "mode should be fixed or dynamic"
        if self.mode == "dynamic":
            self.max_ep_size = min(max_ep_size, self.world_size)
        else:
            assert (
                fixed_dp_size > 0 and fixed_ep_size > 0 and fixed_pp_size > 0
            ), "dp_size, ep_size and pp_size should be greater than 0"
            assert (
                isinstance(fixed_dp_size, int) and isinstance(fixed_ep_size, int) and isinstance(fixed_pp_size, int)
            ), "dp_size, ep_size and pp_size should be int"
            self.ep_size = fixed_ep_size
            self.dp_size = fixed_dp_size
            self.pp_size = fixed_pp_size

        self.has_setup = True

    def get_info(self, num_experts: int, use_tp: bool = False) -> Tuple[int, MoeParallelInfo]:
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

        if self.mode == "dynamic":
            gt_flag = num_experts % self.max_ep_size == 0  # check whether num_experts is greater
            lt_flag = self.max_ep_size % num_experts == 0  # check whether num_experts is less
            assert gt_flag or lt_flag, (
                "Automatic experts placement dose not not support expert number"
                " is not a multiple of ep size or vice versa."
            )
            dp_size = 1 if gt_flag else self.world_size // num_experts
            ep_size = min(self.world_size // dp_size, self.max_ep_size)
            dp_size = self.world_size // ep_size
            pp_size = 1
        else:
            dp_size = self.dp_size
            ep_size = self.ep_size
            pp_size = self.pp_size

        # Calculate the number of experts for each GPU
        if use_tp:
            num_local_experts = num_experts
        else:
            if self.mode == "dynamic":
                num_local_experts = 1 if lt_flag else num_experts // self.max_ep_size
            else:
                num_local_experts = num_experts // ep_size

        if not (ep_size in self.parallel_info_dict):
            self.parallel_info_dict[ep_size] = get_moe_info(ep_size, dp_size, pp_size, ep_inside=self.use_ep_inside)
            if dist.get_rank() == 0:
                if self.use_ep_inside:
                    print(f"MoE Parallel: pp {pp_size}, dp {dp_size}, ep {ep_size}")
                else:
                    print(f"MoE Parallel: pp {pp_size}, ep {ep_size}, dp {dp_size}")

        return num_local_experts, self.parallel_info_dict[ep_size]

    def reset_loss(self):
        self.router_aux_loss, self.router_z_loss = [], []

    def add_loss(self, aux_loss: float = 0.0, z_loss: float = 0.0):
        self.router_aux_loss.append(aux_loss)
        self.router_z_loss.append(z_loss)

    def get_loss(self):
        cur_loss = self.router_aux_loss, self.router_z_loss
        return cur_loss

    def get_parallel(self):
        return self.parallel


MOE_MANAGER = MoEManager()
