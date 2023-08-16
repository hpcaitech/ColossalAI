import random
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn import Module, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.checkpoint_io import CheckpointIO
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.zero.low_level import LowLevelZeroOptimizer

from .pp_plugin_base import PipelinePluginBase

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2


class HybridParallelModule(ModelWrapper):

    def __init__(self, module: Module, precision: str, shard_config: ShardConfig, dp_group: ProcessGroup, use_ddp: bool,
                 ddp_config: dict) -> None:
        self.stage_manager = shard_config.pipeline_stage_manager
        self.dp_group = dp_group
        shardformer = ShardFormer(shard_config)
        module, self.shared_params = shardformer.optimize(module)
        # TODO(ver217): add input type cast
        self.shared_param_process_groups = []
        for shared_param in self.shared_params:
            if len(shared_param) > 0:
                self.shared_param_process_groups.append(
                    self.stage_manager.init_process_group_by_stages(list(shared_param.keys())))
        if precision == 'fp16':
            module = module.half().cuda()
        elif precision == 'bf16':
            module = module.to(dtype=torch.bfloat16).cuda()
        else:
            module = module.cuda()    # train without AMP

        if use_ddp:

            # convert model to sync bn
            module = SyncBatchNorm.convert_sync_batchnorm(module, dp_group)

            # wrap the model with PyTorch DDP
            module = DDP(module, process_group=dp_group, **ddp_config)

        super().__init__(module)

    def sync_shared_params(self):
        for shared_param, group in zip(self.shared_params, self.shared_param_process_groups):
            if self.stage_manager.stage in shared_param:
                param = shared_param[self.stage_manager.stage]
                dist.all_reduce(param.grad, group=group)
            dist.barrier()

    def no_sync(self) -> Iterator[None]:
        # no sync grads across data parallel
        return nullcontext()

    def sync_grads(self):
        # sync grad across data parallel
        if self.dp_group.size() == 1:
            return
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, group=self.dp_group)
                p.grad.div_(self.dp_group.size())

    def unwrap(self):
        module = super().unwrap()
        if isinstance(module, DDP):
            module = module.module
        return module


def init_pipeline_optimizer(optim: Optimizer, model: Module):
    params = set(model.parameters())
    new_param_groups = []
    for group in optim.param_groups:
        params = [p for p in group['params'] if p in params]
        new_param_groups.append({**group, 'params': params})
    optim.__setstate__({'param_groups': new_param_groups})


class HybridParallelNaiveOptimizer(OptimizerWrapper):

    def __init__(self, optim: Optimizer, model: Module, use_pipeline: bool):
        if use_pipeline:
            init_pipeline_optimizer(optim, model)
        super().__init__(optim)


class HybridParallelAMPOptimizer(MixedPrecisionOptimizer):

    def __init__(self,
                 optim: Optimizer,
                 model: Module,
                 use_pipeline: bool,
                 precision: str = 'fp16',
                 initial_scale: float = 2**16,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32,
                 max_norm: float = 0):
        if use_pipeline:
            init_pipeline_optimizer(optim, model)
        super().__init__(optim, precision, initial_scale, min_scale, growth_factor, backoff_factor, growth_interval,
                         hysteresis, max_scale, max_norm)


class HybridParallelZeroOptimizer(LowLevelZeroOptimizer):

    def __init__(
            self,
            optimizer: Optimizer,
            model: Module,
            use_pipeline: bool,
            initial_scale: int = 2**16,    # grad scaler config
            min_scale: int = 1,
            growth_factor: float = 2.,
            backoff_factor: float = .5,
            growth_interval: int = 2000,
            hysteresis: int = 2,
            max_scale: int = 2**24,
            clip_grad_norm: float = 0.0,    # grad clipping
            verbose: bool = False,
            reduce_bucket_size: int = 1024 * 1024,    # communication
            communication_dtype: Optional[torch.dtype] = None,
            overlap_communication: bool = True,
            partition_grad: bool = False,    # stage 2 flag
            cpu_offload: bool = False,    # cpu offload
            dp_process_group: Optional[ProcessGroup] = None,    # the dp pg for comm
            tp_process_group: Optional[ProcessGroup] = None,    # if using tp
            forced_dtype: Optional[torch.dtype] = None):
        if use_pipeline:
            init_pipeline_optimizer(optimizer, model)
        super().__init__(optimizer, initial_scale, min_scale, growth_factor, backoff_factor, growth_interval,
                         hysteresis, max_scale, clip_grad_norm, verbose, reduce_bucket_size, communication_dtype,
                         overlap_communication, partition_grad, cpu_offload, dp_process_group, tp_process_group,
                         forced_dtype)


class HybridParallelPlugin(PipelinePluginBase):
    """
    Plugin for Hybrid Parallel Training.
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
        tp_size (int): The size of tensor parallelism. Tensor parallelism will not be used when tp_size is set to 1.
        pp_size (int): The number of pipeline stages in pipeline parallelism. Pipeline parallelism will not be used when pp_size is set to 1.
        precision (str, optional): Specifies the precision of parameters during training.
                                    Auto-mixied precision will be used when this argument is set to 'fp16' or 'bf16', otherwise model is trained with 'fp32'.
                                    Defaults to 'fp16'.
        zero_stage (int, optional): The stage of ZeRO for data parallelism. Can only be choosed from [0, 1, 2].
                                        When set to 0, ZeRO will not be used. Defaults to 0.
        cpu_offload (bool, optional): Whether to open cpu_offload when using ZeRO. Defaults to False.
        enable_all_optimization (bool, optional): Whether to switch on all the optimizations supported by Shardformer.
                                                    Currently all the optimization methods include fused normalization, flash attention and JIT.
                                                    Defaults to False.
        enable_fused_normalization (bool, optional): Whether to switch on fused normalization. Defaults to False.
        enable_flash_attention (bool, optional): Whether to switch on flash attention. Defaults to False.
        enable_jit_fused (bool, optional): Whether to switch on JIT. Default to Falase.
        num_microbatches (int, optional): Number of microbatches when using pipeline parallelism. Defaults to None.
        initial_scale (float, optional): The initial loss scale of AMP. Defaults to 2**16.
        min_scale (float, optional): The minimum loss scale of AMP. Defaults to 1.
        growth_factor (float, optional): The multiplication factor for increasing loss scale when using AMP. Defaults to 2.
        backoff_factor (float, optional): The multiplication factor for decreasing loss scale when using AMP. Defaults to 0.5.
        growth_interval (int, optional): The number of steps to increase loss scale when no overflow occurs when using AMP. Defaults to 1000.
        hysteresis (int, optional):  The number of overflows before decreasing loss scale when using AMP. Defaults to 2.
        max_scale (float, optional): The maximum loss scale of AMP. Defaults to 2**32.
        max_norm (float, optional): Maximum norm for gradient clipping. Defaults to 0.
        broadcast_buffers (bool, optional): Whether to broadcast buffers in the beginning of training. Only for usage of DDP. Defaults to True.
        bucket_cap_mb (int, optional): The bucket size in MB. Only for usage of DDP. Defaults to 25.
        find_unused_parameters (bool, optional): Whether to find unused parameters. Only for usage of DDP. Defaults to False.
        check_reduction (bool, optional): Whether to check reduction. Only for usage of DDP. Defaults to False.
        gradient_as_bucket_view (bool, optional): Whether to use gradient as bucket view. Only for usage of DDP. Defaults to False.
        static_graph (bool, optional): Whether to use static graph. Only for usage of DDP. Defaults to False.
    """

    def __init__(self,
                 tp_size: int,
                 pp_size: int,
                 precision: str = 'fp16',
                 zero_stage: int = 0,
                 cpu_offload: bool = False,
                 enable_all_optimization: bool = False,
                 enable_fused_normalization: bool = False,
                 enable_flash_attention: bool = False,
                 enable_jit_fused: bool = False,
                 enable_sequence_parallelism: bool = False,
                 num_microbatches: Optional[int] = None,
                 initial_scale: float = 2**16,
                 min_scale: float = 1,
                 growth_factor: float = 2,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 1000,
                 hysteresis: int = 2,
                 max_scale: float = 2**32,
                 max_norm: float = 0,
                 broadcast_buffers=True,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False,
                 gradient_as_bucket_view=False,
                 static_graph=False) -> None:

        super().__init__()
        assert dist.get_world_size() % (
            tp_size * pp_size
        ) == 0, f'world size {dist.get_world_size()} is not divisible by tp_size {tp_size} * pp_size {pp_size}'
        # TODO(ver217): support zero
        assert zero_stage == 0, 'zero is not support yet'
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = dist.get_world_size() // (tp_size * pp_size)
        self.precision = precision
        self.zero_stage = zero_stage
        self.cpu_offload = cpu_offload
        self.enable_all_optimization = enable_all_optimization
        self.enable_fused_normalization = enable_fused_normalization
        self.enable_flash_attention = enable_flash_attention
        self.enable_jit_fused = enable_jit_fused
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.pg_mesh = ProcessGroupMesh(self.dp_size, self.pp_size, self.tp_size)
        self.stage_manager = None
        self.schedule = None
        assert zero_stage in (0, 1, 2)
        if self.pp_size > 1:
            assert num_microbatches is not None, 'num_microbatches must be specified when using pipeline parallelism'
            assert self.zero_stage <= 1, 'zero stage must be 0 or 1 when using pipeline parallelism'
            self.stage_manager = PipelineStageManager(self.pg_mesh, PP_AXIS)
            self.schedule = OneForwardOneBackwardSchedule(num_microbatches, self.stage_manager)
        self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS)
        self.dp_group = self.pg_mesh.get_group_along_axis(DP_AXIS)
        self.shard_config = ShardConfig(tensor_parallel_process_group=self.tp_group,
                                        pipeline_stage_manager=self.stage_manager,
                                        enable_tensor_parallelism=self.tp_size > 1,
                                        enable_all_optimization=self.enable_all_optimization,
                                        enable_fused_normalization=self.enable_fused_normalization,
                                        enable_flash_attention=self.enable_flash_attention,
                                        enable_jit_fused=self.enable_jit_fused,
                                        enable_sequence_parallelism=enable_sequence_parallelism)
        self.amp_config = dict(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        self.ddp_config = dict(broadcast_buffers=broadcast_buffers,
                               bucket_cap_mb=bucket_cap_mb,
                               find_unused_parameters=find_unused_parameters,
                               check_reduction=check_reduction,
                               gradient_as_bucket_view=gradient_as_bucket_view,
                               static_graph=static_graph)
        self.max_norm = max_norm

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return self.pp_size > 1

    def supported_devices(self) -> List[str]:
        return ['cuda']

    def supported_precisions(self) -> List[str]:
        return ['fp16', 'bf16', 'fp32']

    def control_device(self) -> bool:
        return True

    def control_precision(self) -> bool:
        return True

    def support_no_sync(self) -> bool:
        return False

    def control_checkpoint_io(self) -> bool:
        return True

    def configure(
        self,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        if not isinstance(model, ModelWrapper):
            use_ddp = self.dp_size > 1 and self.pp_size == 1 and self.zero_stage == 0
            model = HybridParallelModule(model, self.precision, self.shard_config, self.dp_group, use_ddp,
                                         self.ddp_config)
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            if self.zero_stage == 0:
                if self.precision in ['fp16', 'bf16']:
                    optimizer = HybridParallelAMPOptimizer(optimizer,
                                                           model,
                                                           use_pipeline=self.enable_pipeline_parallelism,
                                                           precision=self.precision,
                                                           max_norm=self.max_norm,
                                                           **self.amp_config)
                else:
                    optimizer = HybridParallelNaiveOptimizer(optimizer,
                                                             model,
                                                             use_pipeline=self.enable_pipeline_parallelism)
            else:
                optimizer = HybridParallelZeroOptimizer(optimizer,
                                                        model,
                                                        use_pipeline=self.enable_pipeline_parallelism,
                                                        partition_grad=(self.zero_stage == 2),
                                                        cpu_offload=self.cpu_offload,
                                                        dp_process_group=self.dp_group,
                                                        tp_process_group=self.tp_group,
                                                        verbose=True,
                                                        clip_grad_norm=self.max_norm,
                                                        **self.amp_config)
        return model, optimizer, criterion, dataloader, lr_scheduler

    def execute_pipeline(self,
                         data_iter: Iterator,
                         model: HybridParallelModule,
                         criterion: Callable[[Any, Any], torch.Tensor],
                         optimizer: Union[HybridParallelNaiveOptimizer, HybridParallelAMPOptimizer,
                                          HybridParallelZeroOptimizer],
                         return_loss: bool = True,
                         return_outputs: bool = False) -> dict:
        assert self.enable_pipeline_parallelism, 'pipeline parallelism is not enabled'
        # return loss or outputs if needed
        ctx = optimizer.no_sync() if isinstance(optimizer, HybridParallelZeroOptimizer) else model.no_sync()
        with ctx:
            outputs = self.schedule.forward_backward_step(model, optimizer, data_iter, criterion, return_loss,
                                                          return_outputs)
        model.sync_shared_params()
        if isinstance(optimizer, HybridParallelZeroOptimizer):
            optimizer.sync_grad()
        else:
            model.sync_grads()
        return outputs

    def prepare_dataloader(self,
                           dataset,
                           batch_size,
                           shuffle=False,
                           seed=1024,
                           drop_last=False,
                           pin_memory=False,
                           num_workers=0,
                           **kwargs):
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
        sampler = DistributedSampler(dataset,
                                     num_replicas=self.pg_mesh.size(DP_AXIS),
                                     rank=self.pg_mesh.coordinate(DP_AXIS),
                                     shuffle=shuffle)

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          **_kwargs)

    def get_checkpoint_io(self) -> CheckpointIO:
        return None

    def no_sync(self, model: Module) -> Iterator[None]:
        raise NotImplementedError
