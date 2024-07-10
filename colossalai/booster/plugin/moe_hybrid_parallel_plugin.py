import random
import warnings
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
from torch.utils.data.distributed import DistributedSampler

from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelModule,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    get_param_info,
    init_pipeline_optimizer,
)
from colossalai.checkpoint_io import MoECheckpointIO
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.policies.base_policy import Policy
from colossalai.tensor.moe_tensor.api import is_moe_tensor
from colossalai.zero.low_level import LowLevelZeroOptimizer


class MoeHybridParallelZeroOptimizer(LowLevelZeroOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        model: Module,
        use_pipeline: bool,
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
        dp_process_group: Optional[ProcessGroup] = None,  # the dp pg for comm
        tp_process_group: Optional[ProcessGroup] = None,  # if using tp
        pp_process_group: Optional[ProcessGroup] = None,
        forced_dtype: Optional[torch.dtype] = None,
        moe_extra_dp_process_group: Optional[ProcessGroup] = None,
    ):
        self.param_info = param_info
        self.stage_manager = model.stage_manager
        self.shared_params = model.shared_params
        self.dp_pg = dp_process_group
        self.tp_pg = tp_process_group
        self.pp_pg = pp_process_group
        if use_pipeline:
            init_pipeline_optimizer(optimizer, model)

        pg_param_list = {
            dp_process_group: [],
            moe_extra_dp_process_group: [],
        }
        for param in model.parameters():
            if is_moe_tensor(param):
                pg_param_list[moe_extra_dp_process_group].append(param)
            else:
                pg_param_list[dp_process_group].append(param)

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
    Plugin for Moe Hybrid Parallel Training.
    Tensor parallel, pipeline parallel and data parallel(DDP/ZeRO) can be picked and combined in this plugin.
    The size of tp and pp should be passed in by user, then the size of dp is automatically calculated from dp_size = world_size / (tp_size * pp_size).

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import HybridParallelPlugin

        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin =  HybridParallelPlugin(tp_size=2, pp_size=2)

        >>> train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, criterion, train_dataloader, _ = booster.boost(model, optimizer, criterion, train_dataloader)

    Args:
        pp_size (int): The number of pipeline stages in pipeline parallelism. Pipeline parallelism will not be used when pp_size is set to 1.
        tp_size (int): The size of tensor parallelism. Tensor parallelism will not be used when tp_size is set to 1.
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
        enable_sequence_overlap (bool): Whether to turn on sequence overlap in Shardformer. Defaults to False.
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
        use_ep_inside (bool, Optional): Whether to use ep inside dp (intra-node) for moe params.
    """

    def __init__(
        self,
        pp_size: int,
        ep_size: int,
        tp_size: int = 1,
        sp_size: int = 1,
        precision: str = "fp16",
        zero_stage: int = 0,
        enable_all_optimization: bool = False,
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        enable_sequence_parallelism: bool = False,
        enable_sequence_overlap: bool = False,
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
        overlap_communication: bool = True,
        use_ep_inside: bool = True,
        custom_policy: Policy = None,
        checkpoint_io: Optional[MoECheckpointIO] = None,
    ) -> None:
        world_size = dist.get_world_size()
        assert tp_size == 1, "Tensor parallel is not supported in MoE yet"
        assert sp_size == 1 and enable_sequence_parallelism is False, "Sequence parallelism it not supported in MoE yet"

        assert (
            world_size % (tp_size * pp_size) == 0
        ), f"world size {world_size} is not divisible by tp_size {tp_size} * pp_size {pp_size}"
        assert (
            world_size % (tp_size * pp_size * ep_size) == 0
        ), f"world size {world_size} is not divisible by tp_size {tp_size} * pp_size {pp_size} * ep_size {ep_size}"

        self.dp_size = world_size // (tp_size * pp_size)
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.sp_size = sp_size
        self.precision = precision
        self.zero_stage = zero_stage
        self.cpu_offload = cpu_offload
        self.enable_all_optimization = enable_all_optimization
        self.enable_fused_normalization = enable_fused_normalization
        self.enable_flash_attention = enable_flash_attention
        self.enable_jit_fused = enable_jit_fused
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.checkpoint_io = checkpoint_io

        logger = get_dist_logger()

        # NOTE: Two process meshes: global dp for non-moe param; dp + ep for moe param
        # See https://hpc-ai.com/blog/enhanced-moe-parallelism-open-source-moe-model-training-can-be-9-times-more-efficient
        # we change pg mesh to (pp, dp, tp) for better moe performance
        assert (
            self.ep_size <= self.dp_size
        ), f"Not enough devices({self.dp_size}) for expert parallelism size({self.ep_size})."

        self.moe_dp_size = self.dp_size // self.ep_size
        self.use_ep_inside = use_ep_inside
        if self.use_ep_inside:
            logger.info(f"MoE Parallel use ep inside dp.", ranks=[0])
            self.pp_axis, self.dp_axis, self.ep_axis, self.tp_axis = 0, 1, 2, 3
            self.pg_mesh = ProcessGroupMesh(self.pp_size, self.moe_dp_size, ep_size, tp_size)
        else:
            logger.info(f"MoE Parallel use ep outside dp.", ranks=[0])
            warnings.warn("Using ep outside dp (cross-node) is strongly discouraged due to communication costs.")
            self.pp_axis, self.dp_axis, self.ep_axis, self.tp_axis = 0, 2, 1, 3
            self.pg_mesh = ProcessGroupMesh(self.pp_size, ep_size, self.moe_dp_size, tp_size)

        self.moe_dp_group = self.pg_mesh.get_group_along_axis(self.dp_axis)
        self.ep_group = self.pg_mesh.get_group_along_axis(self.ep_axis)
        logger.info(f"Non-MoE Parameter Parallel: pp {self.pp_size}, dp {self.dp_size}, tp {tp_size}", ranks=[0])
        logger.info(
            f"MoE Parallel: pp {self.pp_size}, ep {ep_size}, moe dp {self.moe_dp_size}, tp {tp_size}", ranks=[0]
        )

        self.tp_group = self.pg_mesh.get_group_along_axis(
            self.tp_axis
        )  # TODO: support custom tp size for mixtral lm head
        self.global_dp_group = self.pg_mesh.get_group_along_axis((self.dp_axis, self.ep_axis))
        self.pp_group = self.pg_mesh.get_group_along_axis(self.pp_axis)
        # TODO: Currently moe only support partially sequence parallel
        self.sp_group = self.pg_mesh.get_group_along_axis(self.tp_axis)

        self.custom_policy = custom_policy
        self.stage_manager = None
        self.schedule = None

        assert zero_stage in (0, 1, 2)
        if self.pp_size > 1:
            assert (
                num_microbatches is not None or microbatch_size is not None
            ), "num_microbatches or microbatch_size must be specified when using pipeline parallelism"
            assert self.zero_stage <= 1, "zero stage must be 0 or 1 when using pipeline parallelism"
            self.stage_manager = PipelineStageManager(self.pg_mesh, self.pp_axis)
            self.schedule = OneForwardOneBackwardSchedule(
                self.stage_manager, num_microbatches=num_microbatches, microbatch_size=microbatch_size
            )

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=self.tp_group,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=self.tp_size > 1,
            enable_all_optimization=self.enable_all_optimization,
            enable_fused_normalization=self.enable_fused_normalization,
            enable_flash_attention=self.enable_flash_attention,
            enable_jit_fused=self.enable_jit_fused,
            enable_sequence_parallelism=enable_sequence_parallelism,
            enable_sequence_overlap=enable_sequence_overlap,
            ep_group=self.ep_group,
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
        )

        self.max_norm = max_norm

    def prepare_dataloader(
        self, dataset, batch_size, shuffle=False, seed=1024, drop_last=False, pin_memory=False, num_workers=0, **kwargs
    ):
        r"""
        Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader` and `torch.utils.data.DistributedSampler`.


        Args:
            dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random worker seed for sampling, defaults to 1024.
            add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
            drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
                is not divisible by the batch size. If False and the size of dataset is not divisible by
                the batch size, then the last batch will be smaller, defaults to False.
            pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
            num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
            kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                    `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

        Returns:
            :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
        """
        _kwargs = kwargs.copy()
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dp_size,
            rank=dist.get_rank(self.global_dp_group),
            shuffle=shuffle,
        )

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )

    def get_checkpoint_io(self) -> MoECheckpointIO:
        if self.checkpoint_io is None:
            self.checkpoint_io = MoECheckpointIO(
                self.global_dp_group, self.pp_group, self.tp_group, self.ep_group, self.moe_dp_group, self.zero_stage
            )
        else:
            self.checkpoint_io = self.checkpoint_io(
                self.global_dp_group,
                self.pp_group,
                self.tp_group,
                ep_group=self.ep_group,
                moe_dp_group=self.moe_dp_group,
                zero_stage=self.zero_stage,
            )
            if hasattr(self.checkpoint_io, "moe_info"):
                self.checkpoint_io.moe_info = self.moe_info
        return self.checkpoint_io

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
            use_ddp = self.dp_size > 1 and self.pp_size == 1 and self.zero_stage == 0
            model = HybridParallelModule(
                module=model,
                precision=self.precision,
                shard_config=self.shard_config,
                dp_group=self.global_dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
                use_ddp=use_ddp,
                ddp_config=self.ddp_config,
                custom_policy=self.custom_policy,
            )
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
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
                    dp_process_group=self.global_dp_group,
                    tp_process_group=self.tp_group,
                    pp_process_group=self.pp_group,
                    moe_extra_dp_process_group=self.moe_dp_group,
                    verbose=True,
                    clip_grad_norm=self.max_norm,
                    **self.zero_config,
                    **self.amp_config,
                )
            # inject update_master_params
            model.update_master_params = MethodType(optimizer.update_master_params, model)

        return model, optimizer, criterion, dataloader, lr_scheduler
