from collections import defaultdict
from types import MethodType
from typing import Callable, List, Optional, OrderedDict, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.booster.plugin.hybrid_parallel_plugin import (
    PRECISION_TORCH_TYPE,
    SUPPORT_SP_MODE,
    HybridParallelAMPOptimizer,
    HybridParallelModule,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    HybridParallelZeroOptimizer,
    get_param_info,
    reinitialize_optimizer,
)
from colossalai.checkpoint_io import MoECheckpointIO
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.interface.optimizer import DistributedOptim
from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import cast_to_distributed
from colossalai.pipeline.schedule.interleaved_pp import InterleavedSchedule
from colossalai.pipeline.schedule.one_f_one_b import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.shard.grad_ckpt_config import GradientCheckpointConfig
from colossalai.shardformer.shard.shard_config import ShardConfig
from colossalai.tensor.moe_tensor.api import is_moe_tensor


class MoeHybridParallelZeroOptimizer(HybridParallelZeroOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        model: Module,
        use_pipeline: bool,
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
        overlap_allgather: bool = False,
    ):
        if dp_process_group is moe_dp_group:
            pg_param_list = {
                dp_process_group: list(model.parameters()),
            }
        else:
            pg_param_list = {
                dp_process_group: list(filter(lambda p: not is_moe_tensor(p), model.parameters())),
                moe_dp_group: list(filter(is_moe_tensor, model.parameters())),
            }

        if len(pg_param_list[moe_dp_group]) == 0:
            raise ValueError("No parameters found in moe_dp_group, please consider using HybridParallelPlugin instead")

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
            overlap_allgather=overlap_allgather,
        )


class MoeHybridParallelPlugin(HybridParallelPlugin):
    """
    Plugin for MoE Hybrid Parallel Training, which is similar to HybridParallelPlugin
    Tensor parallel, pipeline parallel and data parallel(DDP/ZeRO) can be picked and combined in this plugin.
    The size of tp and pp should be passed in by user, then the size of dp is automatically calculated from dp_size = world_size / (tp_size * pp_size).

    ```python
    from colossalai.booster import Booster
    from colossalai.booster.plugin import MoeHybridParallelPlugin

    model, train_dataset, optimizer, criterion = ...
    plugin =  MoeHybridParallelPlugin(tp_size=2, pp_size=2, ep_size=2)

    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
    booster = Booster(plugin=plugin)
    model, optimizer, criterion, train_dataloader, _ = booster.boost(model, optimizer, criterion, train_dataloader)
    ```

    Args:
        tp_size (int): The size of tensor parallelism. Tensor parallelism will not be used when tp_size is set to 1.
        pp_size (int): The number of pipeline stages in pipeline parallelism. Pipeline parallelism will not be used when pp_size is set to 1.
        ep_size (int): The size of expert parallelism
        sp_size (int): The size of sequence parallelism.
        precision (str, optional): Specifies the precision of parameters during training.
                                    Auto-mixied precision will be used when this argument is set to 'fp16' or 'bf16', otherwise model is trained with 'fp32'.
                                    Defaults to 'fp16'.
        zero_stage (int, optional): The stage of ZeRO for data parallelism. Can only be choosed from [0, 1, 2].
                                        When set to 0, ZeRO will not be used. Defaults to 0.
        enable_all_optimization (bool, optional): Whether to switch on all the optimizations supported by Shardformer.
                                                    Currently all the optimization methods include fused normalization, flash attention and JIT.
                                                    Defaults to False.
        enable_fused_normalization (bool, optional): Whether to switch on fused normalization in Shardformer. Defaults to False.
        enable_flash_attention (bool, optional): Whether to switch on flash attention in Shardformer. Defaults to False.
        enable_jit_fused (bool, optional): Whether to switch on JIT in Shardformer. Default to False.
        enable_sequence_parallelism (bool): Whether to turn on sequence parallelism in Shardformer. Defaults to False.
        sequence_parallelism_mode (str): The Sequence parallelism mode. Can only be choosed from ["split_gather", "ring", "all_to_all"]. Defaults to "split_gather".
        enable_sequence_overlap (bool): Whether to turn on sequence overlap in Shardformer. Defaults to False.
        parallel_output (bool): Whether to keep the output parallel when enabling tensor parallelism. Default to True.
        num_microbatches (int, optional): Number of microbatches when using pipeline parallelism. Defaults to None.
        microbatch_size (int, optional): Microbatch size when using pipeline parallelism.
            Either ``num_microbatches`` or ``microbatch_size`` should be provided if using pipeline.
            If ``num_microbatches`` is provided, this will be ignored. Defaults to None.
        initial_scale (float, optional): The initial loss scale of AMP. Defaults to 2**16.
        min_scale (float, optional): The minimum loss scale of AMP. Defaults to 1.
        growth_factor (float, optional): The multiplication factor for increasing loss scale when using AMP. Defaults to 2.
        backoff_factor (float, optional): The multiplication factor for decreasing loss scale when using AMP. Defaults to 0.5.
        growth_interval (int, optional): The number of steps to increase loss scale when no overflow occurs when using AMP. Defaults to 1000.
        hysteresis (int, optional):  The number of overflows before decreasing loss scale when using AMP. Defaults to 2.
        max_scale (float, optional): The maximum loss scale of AMP. Defaults to 2**32.
        max_norm (float, optional): Maximum norm for gradient clipping. Defaults to 0.
        broadcast_buffers (bool, optional): Whether to broadcast buffers in the beginning of training when using DDP. Defaults to True.
        ddp_bucket_cap_mb (int, optional): The bucket size in MB when using DDP. Defaults to 25.
        find_unused_parameters (bool, optional): Whether to find unused parameters when using DDP. Defaults to False.
        check_reduction (bool, optional): Whether to check reduction when using DDP. Defaults to False.
        gradient_as_bucket_view (bool, optional): Whether to use gradient as bucket view when using DDP. Defaults to False.
        static_graph (bool, optional): Whether to use static graph when using DDP. Defaults to False.
        zero_bucket_size_in_m (int, optional): Gradient reduce bucket size in million elements when using ZeRO. Defaults to 12.
        cpu_offload (bool, optional): Whether to open cpu_offload when using ZeRO. Defaults to False.
        communication_dtype (torch.dtype, optional): Communication dtype when using ZeRO. If not specified, the dtype of param will be used. Defaults to None.
        overlap_communication (bool, optional): Whether to overlap communication and computation when using ZeRO. Defaults to True.
        custom_policy (Policy, optional): Custom policy for Shardformer. Defaults to None.
        pp_style (str, optional): The style for pipeline parallelism. Defaults to '1f1b'.
        num_model_chunks (int, optional): The number of model chunks for interleaved pipeline parallelism. Defaults to 1.
        gradient_checkpoint_config (GradientCheckpointConfig, optional): Configuration for gradient checkpointing. Defaults to None.
        enable_metadata_cache (bool, optional): Whether to enable metadata cache for pipeline parallelism. Defaults to True.
        make_vocab_size_divisible_by (int, optional): it's used when padding the vocabulary size, to make it choose an faster kenel. Default to 64.
        overlap_p2p (bool, optional): Whether to overlap the p2p communication in pipeline parallelism.
        use_fp8 (bool, optional): Whether to enable fp8 mixed precision training. Defaults to False.
        fp8_communication (bool, optional): Whether to enable fp8 communication. Defaults to False.
    """

    def __init__(
        self,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        sp_size: int = None,
        precision: str = "fp16",
        zero_stage: int = 0,
        enable_all_optimization: bool = False,
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        enable_sequence_parallelism: bool = False,
        sequence_parallelism_mode: str = None,
        enable_sequence_overlap: bool = False,
        parallel_output: bool = True,
        num_microbatches: Optional[int] = None,
        microbatch_size: Optional[int] = None,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0,
        broadcast_buffers: bool = True,
        ddp_bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        zero_bucket_size_in_m: int = 12,
        cpu_offload: bool = False,
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = False,
        custom_policy: Policy = None,
        pp_style: str = "1f1b",
        num_model_chunks: int = 1,
        num_layers_per_stage: Optional[List[int]] = None,
        gradient_checkpoint_config: Optional[GradientCheckpointConfig] = None,
        enable_metadata_cache: bool = True,
        make_vocab_size_divisible_by: int = 64,
        moe_dp_outside: bool = True,
        overlap_p2p: bool = True,
        overlap_allgather: bool = False,
        fp8_communication: bool = False,
        use_fp8: bool = False,
    ) -> None:
        self.logger = get_dist_logger()
        if overlap_communication or zero_stage == 2:
            overlap_communication = False
            zero_stage = 1
            self.logger.warning(
                f"overlap_communication and zero_stage are set to False and 1 because "
                f"ZeRO-2 or comm overlap cause program hang when some experts are not routed.",
                ranks=[0],
            )

        assert (
            dist.get_world_size() % (tp_size * pp_size) == 0
        ), f"World size {dist.get_world_size()} is not divisible by tp_size {tp_size} * pp_size {pp_size}"
        if enable_sequence_parallelism:
            self.sequence_parallelism_mode = (
                sequence_parallelism_mode if sequence_parallelism_mode is not None else "all_to_all"
            )
            assert (
                self.sequence_parallelism_mode in SUPPORT_SP_MODE
            ), f"Sequence parallelism mode {self.sequence_parallelism_mode} is not in the supported list {SUPPORT_SP_MODE}"
            if self.sequence_parallelism_mode in ["split_gather", "ring"]:
                assert (
                    tp_size > 1
                ), f"Sequence parallelism mode {self.sequence_parallelism_mode} must be enabled when using tensor parallelism"
                if sp_size != 1:
                    self.logger.warning(
                        f"The sp_size will be the same as tp_size in sequence parallelism mode {self.sequence_parallelism_mode},"
                        "will ignore the given sequence parallelism size.",
                        ranks=[0],
                    )
                self.sp_size = 1
                self.dp_size = dist.get_world_size() // (tp_size * pp_size)
            elif self.sequence_parallelism_mode in ["all_to_all"]:
                self.sp_size = 1 if sp_size is None else sp_size
                self.dp_size = dist.get_world_size() // (self.sp_size * pp_size * tp_size)
        else:
            self.dp_size = dist.get_world_size() // (tp_size * pp_size)
            assert (
                sp_size == 1 or sp_size is None
            ), f"You should not set sp_size when sequence parallelism is not enabled."
            self.sp_size = 1

        assert self.dp_size % ep_size == 0, f"dp_size should be divisible by ep_size, {self.dp_size=} {ep_size=}"
        self.moe_dp_size = self.dp_size // ep_size
        self.ep_size = ep_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.precision = precision
        self.zero_stage = zero_stage
        self.cpu_offload = cpu_offload
        self.enable_all_optimization = enable_all_optimization
        self.enable_fused_normalization = enable_fused_normalization
        self.enable_flash_attention = enable_flash_attention
        self.enable_jit_fused = enable_jit_fused
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if moe_dp_outside:
            self.moe_dp_axis, self.pp_axis, self.ep_axis, self.tp_axis, self.sp_axis = 0, 1, 2, 3, 4
            self.pg_mesh = ProcessGroupMesh(self.moe_dp_size, self.pp_size, self.ep_size, self.tp_size, self.sp_size)
        else:
            self.pp_axis, self.moe_dp_axis, self.ep_axis, self.tp_axis, self.sp_axis = 0, 1, 2, 3, 4
            self.pg_mesh = ProcessGroupMesh(self.pp_size, self.moe_dp_size, self.ep_size, self.tp_size, self.sp_size)

        self.stage_manager = None
        self.schedule = None
        self.custom_policy = custom_policy
        assert zero_stage in (0, 1, 2)
        if self.pp_size > 1:
            assert pp_style in ["1f1b", "interleaved"], "Unsupported pipeline parallelism style"
            assert pp_style == "interleaved" or num_model_chunks == 1, "num_model_chunks must be 1 when using 1f1b"
            assert (
                num_microbatches is not None or microbatch_size is not None
            ), "num_microbatches or microbatch_size must be specified when using pipeline parallelism"
            assert (
                self.zero_stage <= 1
            ), "To avoid prohibitive gradient synchronization costs, zero stage must be 0 or 1 when using pipeline parallelism"
            self.stage_manager = PipelineStageManager(
                self.pg_mesh,
                pipeline_axis=self.pp_axis,
                enable_interleave=pp_style == "interleaved",
                num_model_chunks=num_model_chunks,
                num_layers_per_stage=num_layers_per_stage,
            )

            if pp_style == "interleaved":
                assert num_model_chunks > 1, "number of model chunks must be > 1 when using interleaved"
                self.schedule = InterleavedSchedule(
                    stage_manager=self.stage_manager,
                    num_model_chunks=num_model_chunks,
                    num_microbatch=num_microbatches,
                    microbatch_size=microbatch_size,
                    enable_metadata_cache=enable_metadata_cache,
                    overlap_p2p=overlap_p2p,
                )
            elif pp_style == "1f1b":
                self.schedule = OneForwardOneBackwardSchedule(
                    stage_manager=self.stage_manager,
                    num_microbatches=num_microbatches,
                    microbatch_size=microbatch_size,
                    enable_metadata_cache=enable_metadata_cache,
                )
            else:
                raise NotImplementedError()

        self.tp_group = self.pg_mesh.get_group_along_axis(self.tp_axis)
        self.dp_group = self.pg_mesh.get_group_along_axis([self.moe_dp_axis, self.ep_axis])
        self.pp_group = self.pg_mesh.get_group_along_axis(self.pp_axis)
        self.moe_dp_group = self.pg_mesh.get_group_along_axis(self.moe_dp_axis)
        self.ep_group = self.pg_mesh.get_group_along_axis(self.ep_axis)
        if self.enable_sequence_parallelism and self.sequence_parallelism_mode in ["split_gather", "ring"]:
            self.sp_group = self.pg_mesh.get_group_along_axis(self.tp_axis)
        else:
            self.sp_group = self.pg_mesh.get_group_along_axis(self.sp_axis)
        self.use_fp8 = use_fp8

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=self.tp_group,
            sequence_parallel_process_group=self.sp_group,
            ep_group=self.ep_group,
            moe_dp_group=self.moe_dp_group,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=self.tp_size > 1,
            enable_all_optimization=self.enable_all_optimization,
            enable_fused_normalization=self.enable_fused_normalization,
            enable_flash_attention=self.enable_flash_attention,
            enable_jit_fused=self.enable_jit_fused,
            enable_sequence_parallelism=enable_sequence_parallelism,
            sequence_parallelism_mode=sequence_parallelism_mode,
            enable_sequence_overlap=enable_sequence_overlap,
            parallel_output=parallel_output,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by,
            gradient_checkpoint_config=gradient_checkpoint_config,
            fp8_communication=fp8_communication,
        )
        self.amp_config = dict(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        self.ddp_config = dict(
            broadcast_buffers=broadcast_buffers,
            bucket_cap_mb=ddp_bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        self.zero_config = dict(
            reduce_bucket_size=zero_bucket_size_in_m * 1024 * 1024,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
            cpu_offload=cpu_offload,
            partition_grad=(self.zero_stage == 2),
            forced_dtype=PRECISION_TORCH_TYPE[precision],
            overlap_allgather=overlap_allgather,
        )

        self.max_norm = max_norm

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
            use_ddp = (self.dp_size > 1 and self.pp_size == 1 and self.zero_stage == 0) or (
                self.dp_size == 1
                and self.pp_size == 1
                and self.enable_sequence_parallelism
                and self.sequence_parallelism_mode == "all_to_all"
            )

            # sync gradients across DP * SP ranks
            if self.enable_sequence_parallelism and self.sequence_parallelism_mode == "all_to_all":
                dp_group = self.pg_mesh.create_group_along_axis([self.moe_dp_axis, self.ep_axis, self.sp_axis])
            else:
                dp_group = self.dp_group

            if use_ddp:
                self.logger.warning(
                    f"Will have to check all params are used in pytorch DDP since not all experts are always activated",
                    ranks=[0],
                )
                self.ddp_config["find_unused_parameters"] = True

                if dist.get_process_group_ranks(dp_group) != dist.get_process_group_ranks(self.moe_dp_group):
                    raise ValueError(
                        f"if pytorch DDP is used, dp_group and moe_dp_group are expected to be the same since DDP can only reduce grad across a single group, but found dp_group {dist.get_process_group_ranks(dp_group)} and moe_dp_group {dist.get_process_group_ranks(self.moe_dp_group)}, you might want to modify your config to bypass DDP \nhint: check the above ddp condition to by pass this"
                    )

            model = HybridParallelModule(
                module=model,
                precision=self.precision,
                shard_config=self.shard_config,
                dp_group=dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
                use_ddp=use_ddp,
                ddp_config=self.ddp_config,
                custom_policy=self.custom_policy,
                use_fp8=self.use_fp8,
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
                is_zero = True
                if self.dp_size <= 1:
                    self.logger.warning(
                        "Use Zero Optimizer when data parallel size is 1 may introduce unnecessary overhead. "
                        "If you do not intend to use cpu_offload, please consider set zero_stage=0.",
                        ranks=[0],
                    )
                assert self.precision != "fp32", "Please set precision to 'fp16' or 'bf16' when using ZeRO."
                optimizer = MoeHybridParallelZeroOptimizer(
                    optimizer,
                    model,
                    use_pipeline=self.enable_pipeline_parallelism,
                    param_info=param_info,
                    dp_process_group=dp_group,
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
