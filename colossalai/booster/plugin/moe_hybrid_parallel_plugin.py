import warnings
from types import MethodType
from typing import Callable, Optional, OrderedDict, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelModule,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    get_param_info,
    reinitialize_optimizer,
)
from colossalai.checkpoint_io import MoECheckpointIO
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.zero.low_level import LowLevelZeroOptimizer


class MoeHybridParallelZeroOptimizer(LowLevelZeroOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        model: Module,
        use_pipeline: bool,
        dp_process_group: ProcessGroup,  # the dp pg for comm
        moe_dp_group: ProcessGroup,  # the moe dp pg for gomm
        param_info: OrderedDict,
        initial_scale: int = 2**16,  # grad scaler config
        min_scale: int = 1,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        hysteresis: int = 2,
        max_scale: int = 2**24,
        clip_grad_norm: float = 0.0,  # grad clipping
        verbose: bool = False,
        reduce_bucket_size: int = 1024 * 1024,  # communication
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = True,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        forced_dtype: Optional[torch.dtype] = None,
    ):
        self.param_info = param_info
        self.stage_manager = model.stage_manager
        self.shared_params = model.shared_params
        self.dp_pg = dp_process_group

        if use_pipeline:
            reinitialize_optimizer(optimizer, model)

        pg_param_list = {
            dp_process_group: list(filter(lambda p: not is_moe_tensor(p), model.parameters())),
            moe_dp_group: list(filter(is_moe_tensor, model.parameters())),
        }

        super().__init__(
            optimizer=optimizer,
            pg_to_param_list=pg_param_list,
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            clip_grad_norm=clip_grad_norm,
            verbose=verbose,
            reduce_bucket_size=reduce_bucket_size,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
            partition_grad=partition_grad,
            cpu_offload=cpu_offload,
            forced_dtype=forced_dtype,
        )


class MoeHybridParallelPlugin(HybridParallelPlugin):
    """
    TODO: add docstring
    """

    def __init__(self, ep_size: int, ep_tp_size: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.use_ddp = self.dp_size > 1 and self.pp_size == 1 and self.zero_stage == 0
        if self.use_ddp:
            warnings.warn(
                f"Will have to check all params are used in pytorch DDP since not all experts are always activated"
            )
            self.ddp_config["find_unused_parameters"] = True

        if ep_tp_size != 1:
            raise NotImplementedError

        world_size = dist.get_world_size()

        self.moe_dp_size = world_size // (ep_size * ep_tp_size)
        self.ep_size = ep_size
        self.moe_tp_size = ep_tp_size

        self.moe_pg_mesh = ProcessGroupMesh(self.moe_dp_size, self.ep_size, self.moe_tp_size)
        self.moe_dp_axis, self.ep_axis, self.moe_tp_axis = 0, 1, 2

        self.moe_dp_group = self.moe_pg_mesh.get_group_along_axis(self.moe_dp_axis)
        self.ep_group = self.moe_pg_mesh.get_group_along_axis(self.ep_axis)
        self.moe_tp_group = self.moe_pg_mesh.get_group_along_axis(self.moe_tp_axis)

        # set ep_group after super init
        # TODO do it in a better way
        self.shard_config.ep_group = self.ep_group

    def get_checkpoint_io(self) -> MoECheckpointIO:
        return MoECheckpointIO(
            self.dp_group, self.pp_group, self.tp_group, self.ep_group, self.moe_dp_group, self.zero_stage
        )

    def configure(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        param_info = get_param_info(optimizer)
        if not isinstance(model, ModelWrapper):
            model = HybridParallelModule(
                module=model,
                precision=self.precision,
                shard_config=self.shard_config,
                dp_group=self.dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
                use_ddp=self.use_ddp,
                ddp_config=self.ddp_config,
                custom_policy=self.custom_policy,
            )
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            if self.ep_size > 1:
                # if ep is enabled, the num of (moe) paramaters changed since they are sharded among ep groups
                # but the optimizer is not aware of ep, so we need to update the optimizer
                reinitialize_optimizer(optimizer, model)

            if self.zero_stage == 0:
                if self.precision in ["fp16", "bf16"]:
                    optimizer = HybridParallelAMPOptimizer(
                        optimizer,
                        model,
                        use_pipeline=self.enable_pipeline_parallelism,
                        param_info=param_info,
                        precision=self.precision,
                        max_norm=self.max_norm,
                        **self.amp_config,
                    )
                else:
                    optimizer = HybridParallelNaiveOptimizer(
                        optimizer, model, use_pipeline=self.enable_pipeline_parallelism, param_info=param_info
                    )
            else:
                assert self.dp_size > 1, "Please use Zero when data parallel size is greater than 1."
                assert self.precision != "fp32", "Please set precision to 'fp16' or 'bf16' when using ZeRO."
                optimizer = MoeHybridParallelZeroOptimizer(
                    optimizer,
                    model,
                    use_pipeline=self.enable_pipeline_parallelism,
                    param_info=param_info,
                    dp_process_group=self.dp_group,
                    moe_dp_group=self.moe_dp_group,
                    verbose=True,
                    clip_grad_norm=self.max_norm,
                    **self.zero_config,
                    **self.amp_config,
                )
            # inject update_master_params
            model.update_master_params = MethodType(optimizer.update_master_params, model)

        return model, optimizer, criterion, dataloader, lr_scheduler
