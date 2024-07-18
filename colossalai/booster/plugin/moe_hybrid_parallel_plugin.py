import warnings
from collections import defaultdict
from types import MethodType
from typing import Callable, Optional, OrderedDict, Tuple

import numpy as np
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
    HybridParallelZeroOptimizer,
    get_param_info,
    reinitialize_optimizer,
)
from colossalai.checkpoint_io import MoECheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.interface.optimizer import DistributedOptim
from colossalai.nn.optimizer import cast_to_distributed
from colossalai.tensor.moe_tensor.api import is_moe_tensor


class MoeHybridParallelZeroOptimizer(HybridParallelZeroOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        model: Module,
        use_pipeline: bool,
        force_overlap_comm: bool,  # force overlap comm
        dp_process_group: Optional[ProcessGroup],  # the dp pg for comm
        tp_process_group: Optional[ProcessGroup],  # if using tp
        pp_process_group: Optional[ProcessGroup],  # if using pp
        moe_dp_group: ProcessGroup,  # moe dp pg for comm
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
        overlap_communication: bool = False,
        partition_grad: bool = False,  # stage 2 flag
        cpu_offload: bool = False,  # cpu offload
        forced_dtype: Optional[torch.dtype] = None,
    ):
        WARN_STR = "Note that you need to make sure every expert are routed (i.e.) every expert has backward, otherwise this might lead to program hang or inconsistent result"
        if not force_overlap_comm and (overlap_communication or partition_grad):
            raise RuntimeError(
                WARN_STR
                + " If you are not sure about this, set (overlap_communication=False and partition_grad=False) or force_overlap_comm=True"
            )

        if force_overlap_comm:
            overlap_communication = True
            warnings.warn(WARN_STR + " Please make sure of this.")

        pg_param_list = {
            dp_process_group: list(filter(lambda p: not is_moe_tensor(p), model.parameters())),
            moe_dp_group: list(filter(is_moe_tensor, model.parameters())),
        }

        super().__init__(
            model=model,
            optimizer=optimizer,
            use_pipeline=use_pipeline,
            param_info=param_info,
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
            tp_process_group=tp_process_group,
            pp_process_group=pp_process_group,
            forced_dtype=forced_dtype,
            pg_to_param_list=pg_param_list,
        )


class MoeHybridParallelPlugin(HybridParallelPlugin):
    """
    TODO: add docstring
    """

    def __init__(self, ep_size: int, moe_tp_size: int = 1, force_overlap_comm=False, *args, **kwargs) -> None:
        if "overlap_communication" not in kwargs:
            kwargs["overlap_communication"] = False  # default by true in super class

        super().__init__(*args, **kwargs)

        if ep_size <= 1:
            raise ValueError("Use HybridParallelPlugin when ep_size <= 1")

        self.ep_size = ep_size
        self.moe_tp_size = moe_tp_size

        self._init_moe_param_comm()

        self.use_ddp = (self.dp_size > 1 and self.pp_size == 1 and self.zero_stage == 0) or (
            self.dp_size == 1
            and self.pp_size == 1
            and self.enable_sequence_parallelism
            and self.sequence_parallelism_mode == "all_to_all"
        )

        if self.use_ddp:
            warnings.warn(
                f"Will have to check all params are used in pytorch DDP since not all experts are always activated"
            )
            self.ddp_config["find_unused_parameters"] = True

            if dist.get_process_group_ranks(self.dp_group) != dist.get_process_group_ranks(self.moe_dp_group):
                # TODO it might make sense to support non-moe with tp on but moe with tp off
                raise ValueError(
                    f"if ddp is used, dp_group and moe_dp_group are expected to be the same since DDP can only reduce grad across a single group, but found dp_group {dist.get_process_group_ranks(self.dp_group)} and moe_dp_group {dist.get_process_group_ranks(self.moe_dp_group)}, you might want to use HybridParallelPlugin or set zero_stage > 0"
                )

        # set param group in shard config
        self.shard_config.ep_group = self.ep_group
        self.shard_config.moe_dp_group = self.moe_dp_group
        self.shard_config.moe_tp_group = self.moe_tp_group

        self.force_overlap_comm = force_overlap_comm

    def _init_moe_param_comm(self):
        world_size = dist.get_world_size()

        if self.enable_sequence_parallelism:
            # if sequence parallelism is enabled, we reuse the same group for ep and sp
            if self.sequence_parallelism_mode == "all_to_all":
                # when sequence parallelism is enabled, ep_group reuses sp_group
                if self.ep_size != self.sp_size:
                    raise ValueError(
                        f"ep_size={self.ep_size} should be equal to sp_size={self.sp_size} or turned off when sequence parallelism is enabled"
                    )

                # since we are reusing sp_group, moe_dp_group will be derived as dp_group
                self.moe_dp_size = self.dp_size
                self.moe_dp_group = self.dp_group  # NOTE: sequence of value assignment matters
                self.dp_group = self.pg_mesh.create_group_along_axis([self.dp_axis, self.sp_axis])
                self.ep_group = self.sp_group
                self.moe_tp_group = self.tp_group
            else:
                raise NotImplementedError(
                    f"sequence_parallelism_mode={self.sequence_parallelism_mode} is not supported"
                )

        else:
            self.moe_dp_size = world_size // (self.pp_size * self.ep_size * self.moe_tp_size)

            if self.moe_dp_size * self.pp_size * self.ep_size * self.moe_tp_size != world_size:
                raise ValueError(
                    f"world_size={world_size} is not divisible by pp_size={self.pp_size} * moe_dp_size={self.moe_dp_size} * ep_size={self.ep_size} * moe_tp_size={self.moe_tp_size}"
                )

            self.moe_dp_group = None
            self.ep_group = None
            self.moe_tp_group = None

            # create submesh for ep, moe_dp, moe_tp
            ranks_by_pp_stage = self.pg_mesh.get_group_along_axis(
                [self.dp_axis, self.tp_axis, self.sp_axis], return_ranks_by_group=True
            )

            global_rank = self.pg_mesh.rank
            pp_rank = self.pg_mesh.coordinate(self.pp_axis)

            # create groups from submesh
            for stage_idx, stage_rank in enumerate(ranks_by_pp_stage):
                # axis 0 is moe_dp, axis 1 is ep, axis 2 is moe_tp
                submesh = np.array(stage_rank).reshape(self.moe_dp_size, self.ep_size, self.moe_tp_size)

                # hardcode here since we only have 3 axis
                # moe_dp_group
                for ep_idx in range(self.ep_size):
                    for moe_tp_idx in range(self.moe_tp_size):
                        moe_dp_ranks = submesh[:, ep_idx, moe_tp_idx].flatten().tolist()
                        group = dist.new_group(moe_dp_ranks)
                        if pp_rank == stage_idx and global_rank in moe_dp_ranks:
                            assert self.moe_dp_group is None
                            self.moe_dp_group = group
                # ep_group
                for moe_dp_idx in range(self.moe_dp_size):
                    for moe_tp_idx in range(self.moe_tp_size):
                        ep_ranks = submesh[moe_dp_idx, :, moe_tp_idx].flatten().tolist()
                        group = dist.new_group(ep_ranks)
                        if pp_rank == stage_idx and global_rank in ep_ranks:
                            assert self.ep_group is None
                            self.ep_group = group
                # moe_tp_group
                for moe_dp_idx in range(self.moe_dp_size):
                    for ep_idx in range(self.ep_size):
                        moe_tp_ranks = submesh[moe_dp_idx, ep_idx, :].flatten().tolist()
                        group = dist.new_group(moe_tp_ranks)
                        if pp_rank == stage_idx and global_rank in moe_tp_ranks:
                            assert self.moe_tp_group is None
                            self.moe_tp_group = group

        if dist.get_process_group_ranks(self.tp_group) != dist.get_process_group_ranks(self.moe_tp_group):
            # NOTE: different tp settings between moe and non moe param are complex to handle
            # we simply reuse tp_group as moe_tp_group, this implies that dp_size == moe_dp_size * ep_size
            raise NotImplementedError(
                f"Only support shared tp group between moe and non moe params, but found non-moe tp {dist.get_process_group_ranks(self.tp_group)}, moe tp {dist.get_process_group_ranks(self.moe_tp_group)}, please make sure tp_size == moe_tp_size"
            )

        self.logger.info(
            f"{type(self).__name__}: {self.ep_size=} {self.moe_dp_size=} {self.moe_tp_size=} {self.sp_size=}\n"
            f"rank {dist.get_rank()} moe_dp_group {dist.get_process_group_ranks(self.moe_dp_group)} ep_group {dist.get_process_group_ranks(self.ep_group)} moe_tp_group {dist.get_process_group_ranks(self.moe_tp_group)} sp_group {dist.get_process_group_ranks(self.sp_group)}",
            ranks=[0],
        )

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

        # TODO: Support Galore + ZeRO
        # Replace with distributed implementation if exists
        optimizer = cast_to_distributed(optimizer)

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
                is_zero = False
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
                        optimizer,
                        model,
                        use_pipeline=self.enable_pipeline_parallelism,
                        param_info=param_info,
                        max_norm=self.max_norm,
                        pp_process_group=self.pp_group,
                        tp_process_group=self.tp_group,
                    )
            else:
                if self.dp_size <= 1:
                    warnings.warn(
                        "Use Zero Optimizer when data parallel size is 1 may introduce unnecessary overhead. "
                        "If you do not intend to use cpu_offload, please consider set zero_stage=0."
                    )
                assert self.precision != "fp32", "Please set precision to 'fp16' or 'bf16' when using ZeRO."
                optimizer = MoeHybridParallelZeroOptimizer(
                    optimizer,
                    model,
                    use_pipeline=self.enable_pipeline_parallelism,
                    force_overlap_comm=self.force_overlap_comm,
                    param_info=param_info,
                    dp_process_group=self.dp_group,
                    tp_process_group=self.tp_group,
                    pp_process_group=self.pp_group,
                    moe_dp_group=self.moe_dp_group,
                    verbose=True,
                    clip_grad_norm=self.max_norm,
                    **self.zero_config,
                    **self.amp_config,
                )
            # inject update_master_params
            model.update_master_params = MethodType(optimizer.update_master_params, model)

            # Setup optimizers that require global states
            optim = optimizer.optim
            if isinstance(optim, DistributedOptim):
                shard_to_param = optimizer.get_master_to_working_map() if is_zero else {}
                padding_map = optimizer.get_param_padding_map() if is_zero else defaultdict(int)
                optim.setup_distributed(self.tp_group, self.dp_group, shard_to_param, padding_map, is_zero)

        return model, optimizer, criterion, dataloader, lr_scheduler
