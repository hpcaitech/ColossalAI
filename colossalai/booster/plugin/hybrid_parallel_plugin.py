import copy
import logging
import os
import random
from contextlib import nullcontext
from functools import partial, reduce
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Iterator, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.nn import Module, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.amp.naive_amp.mixed_precision_optimizer import MixedPrecisionOptimizer
from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO, GeneralCheckpointIO
from colossalai.checkpoint_io.utils import (
    calculate_tensor_size,
    get_model_base_filenames,
    get_optimizer_base_filenames,
    get_shard_filename,
    is_safetensors_available,
    load_shard_state_dict,
    load_state_dict_into_model,
    save_param_groups,
    save_state_dict,
    save_state_dict_shards,
)
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.tensor.d_tensor import (
    is_customized_distributed_tensor,
    is_distributed_tensor,
    to_global,
    to_global_for_customized_distributed_tensor,
)
from colossalai.zero.low_level import LowLevelZeroOptimizer

from .pp_plugin_base import PipelinePluginBase

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, _IncompatibleKeys
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2


def _convert_floating_point(x, dtype: torch.dtype = torch.float16):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype)
    return x


class HybridParallelModule(ModelWrapper):

    def __init__(self, module: Module, precision: str, shard_config: ShardConfig, dp_group: ProcessGroup, use_ddp: bool,
                 ddp_config: dict) -> None:

        self.stage_manager = shard_config.pipeline_stage_manager
        self.dp_group = dp_group

        shardformer = ShardFormer(shard_config)
        module, self.shared_params = shardformer.optimize(module)

        # setting process groups for shared parameters
        self.shared_param_process_groups = []
        for shared_param in self.shared_params:
            if len(shared_param) > 0:
                self.shared_param_process_groups.append(
                    self.stage_manager.init_process_group_by_stages(list(shared_param.keys())))

        # setting mixed_precision
        self.mixed_precision = None
        if precision == 'fp16':
            self.mixed_precision = torch.float16
        elif precision == 'bf16':
            self.mixed_precision = torch.bfloat16
        if self.mixed_precision is not None:
            module = module.to(self.mixed_precision)
        module = module.cuda()

        # setting input type cast when using mixed precision
        self.convert_fn = None
        if self.mixed_precision is not None:
            self.convert_fn = partial(_convert_floating_point, dtype=self.mixed_precision)

        # setting ddp configs
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

    def forward(self, *args, **kwargs):
        if self.convert_fn is not None:
            args = tree_map(self.convert_fn, args)
            kwargs = tree_map(self.convert_fn, kwargs)
        return super().forward(*args, **kwargs)

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
    """

    def __init__(self,
                 tp_size: int,
                 pp_size: int,
                 precision: str = 'fp16',
                 zero_stage: int = 0,
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
                 broadcast_buffers: bool = True,
                 ddp_bucket_cap_mb: int = 25,
                 find_unused_parameters: bool = False,
                 check_reduction: bool = False,
                 gradient_as_bucket_view: bool = False,
                 static_graph: bool = False,
                 zero_bucket_size_in_m: int = 12,
                 cpu_offload: bool = False,
                 communication_dtype: Optional[torch.dtype] = None,
                 overlap_communication: bool = True) -> None:

        super().__init__()
        assert dist.get_world_size() % (
            tp_size * pp_size
        ) == 0, f'world size {dist.get_world_size()} is not divisible by tp_size {tp_size} * pp_size {pp_size}'

        if enable_sequence_parallelism:
            assert tp_size > 1, 'Sequence parallelism must be enabled when using tensor parallelism'

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
        self.pp_group = self.pg_mesh.get_group_along_axis(PP_AXIS)
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
                               bucket_cap_mb=ddp_bucket_cap_mb,
                               find_unused_parameters=find_unused_parameters,
                               check_reduction=check_reduction,
                               gradient_as_bucket_view=gradient_as_bucket_view,
                               static_graph=static_graph)

        self.zero_config = dict(reduce_bucket_size=zero_bucket_size_in_m * 1024 * 1024,
                                communication_dtype=communication_dtype,
                                overlap_communication=overlap_communication,
                                cpu_offload=cpu_offload,
                                partition_grad=(self.zero_stage == 2))

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
                assert self.dp_size > 1, "Please use Zero when data parallel size is greater than 1."
                assert self.precision != 'fp32', "Please set precision to 'fp16' or 'bf16' when using ZeRO."
                optimizer = HybridParallelZeroOptimizer(optimizer,
                                                        model,
                                                        use_pipeline=self.enable_pipeline_parallelism,
                                                        dp_process_group=self.dp_group,
                                                        tp_process_group=self.tp_group,
                                                        verbose=True,
                                                        clip_grad_norm=self.max_norm,
                                                        **self.zero_config,
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
        return HypridParallelCheckpointIO(self.pg_mesh)

    def no_sync(self, model: Module) -> Iterator[None]:
        raise NotImplementedError


class HypridParallelCheckpointIO(GeneralCheckpointIO):
    """
    CheckpointIO for Hybrid Parallel Training.

    Args:
        pg_mesh (ProcessGroupMesh): Process group mesh containing information of process groups along different dimensions.
    """

    def __init__(self, pg_mesh: ProcessGroupMesh) -> None:
        super().__init__()
        self.dp_group = pg_mesh.get_group_along_axis(DP_AXIS)
        self.pp_group = pg_mesh.get_group_along_axis(PP_AXIS)
        self.tp_group = pg_mesh.get_group_along_axis(TP_AXIS)
        self.dp_rank = dist.get_rank(self.dp_group)
        self.tp_rank = dist.get_rank(self.tp_group)
        self.pp_rank = dist.get_rank(self.pp_group)
        self.dp_size, self.pp_size, self.tp_size = pg_mesh.shape

    @staticmethod
    def _model_sharder(model: nn.Module,
                       prefix: str = '',
                       keep_vars: bool = False,
                       size_per_shard: int = 1024) -> Iterator[Tuple[OrderedDict, int]]:
        # An internel method that breaks state_dict of model into shards within limited size.

        state_dict_sharder = _StateDictSharder(size_per_shard)

        # Save parameters.
        for name, param in model.named_parameters():
            if param is None:
                continue
            # Gather tensor pieces when using tensor parallel.
            param_ = param if keep_vars else param.detach()
            if is_distributed_tensor(param_):
                param_ = to_global(param_)
            elif is_customized_distributed_tensor(param_):
                param_ = to_global_for_customized_distributed_tensor(param_)

            block, block_size = state_dict_sharder.append(prefix + name, param_)
            if block is not None:
                yield block, block_size

        # Save buffers.
        for name, buf in model.named_buffers():
            if buf is not None and name not in model._non_persistent_buffers_set:
                buffer = buf if keep_vars else buf.detach()
                block, block_size = state_dict_sharder.append(prefix + name, buffer)
                if block is not None:
                    yield block, block_size

        # Save extra states.
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(model.__class__, "get_extra_state",
                   torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
            extra_state = model.get_extra_state()
            block, block_size = state_dict_sharder.append(extra_state_key, extra_state)
            if block is not None:
                yield block, block_size

        # Return the last block in sharder.
        yield state_dict_sharder.current_block, state_dict_sharder.current_block_size

    @staticmethod
    def _optimizer_sharder(optimizer: Optimizer, size_per_shard: int = 1024):
        # An internel method that breaks state_dict of optimizer into shards within limited size.
        # TODO (Baizhou): Implement sharding feature of optimizer.
        pass

    def save_sharded_model(self,
                           model: nn.Module,
                           checkpoint: str,
                           gather_dtensor: bool = True,
                           prefix: Optional[str] = None,
                           size_per_shard: int = 1024,
                           use_safetensors: bool = False) -> None:
        """
        Save sharded model checkpoint under the given checkpointing path.
        The following files will be created under the path:
        - An index file (pytorch_optim.bin.index.json) containing a map between model params/buffers and file names.
        - Multiple files that store state tensors of models.
          If pipeline parallelism is used, the filenames are in the form of "pytorch_model.<prefix>-stage-000XX-shard-000XX.bin".
          If pipeline parallelism is not used, "pytorch_model.<prefix>-000XX.bin"


        Args:
            model (nn.Module): Model on local device to be saved.
            checkpoint_path (str): Checkpointing path which should be a directory path.
            gather_dtensor (bool, optional): Whether to gather_dtensor, currently not used. Defaults to True.
            prefix (str, optional): Perfix of file to save. Defaults to None.
            size_per_shard (int, optional): Size per shard in MB. Defaults to 1024.
            use_safetensors (bool, optional): Whether to use safe tensors. Defaults to False.
        """

        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # Devices along the same dp_group share the same copies of model.
        # So only let the device with dp_rank == 0 save the model.
        if self.dp_rank != 0:
            return

        # Then collect the sharded parameters & buffers along tp_group.
        # Only devices with tp_size == 0 are responsible for model saving.
        state_dict_shard = HypridParallelCheckpointIO._model_sharder(model, size_per_shard=size_per_shard)
        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
        index_file = CheckpointIndexFile(checkpoint)
        control_saving = (self.tp_rank == 0)

        if self.pp_size == 1:
            # When pipeline is not used, save the model shards as in general checkpointIO
            total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard,
                                                checkpoint=checkpoint,
                                                index_file=index_file,
                                                base_filename=weights_name,
                                                is_master=control_saving,
                                                use_safetensors=use_safetensors)
            if control_saving:
                index_file.append_meta_data("total_size", total_size)
                index_file.write_index_file(save_index_file)
                logging.info(f"The model is split into checkpoint shards. "
                             f"You can find where each parameters has been saved in the "
                             f"index located at {save_index_file}.")

        else:
            # When pipeline is used, each stage produces its own shard files and index files.
            # Index files belonging to each stage are saved under a temporary folder ./tmp_index_files/
            # After all the state_dicts have been saved, the master rank integrates all the index files into one final index file and deletes the tmp folder.

            final_index_file_path = copy.deepcopy(save_index_file)
            tmp_index_file_folder = os.path.join(checkpoint, "tmp_index_files")
            Path(tmp_index_file_folder).mkdir(parents=True, exist_ok=True)

            # Manage filenames of sharded weights and index file for each pipeline stage.
            weights_name = weights_name.replace(".bin", f"-stage-{self.pp_rank:05d}-shard.bin")
            weights_name = weights_name.replace(".safetensors", f"-stage-{self.pp_rank:05d}-shard.safetensors")
            save_index_file = save_index_file.replace(".json", f"-stage-{self.pp_rank:05d}.json")
            save_index_file = os.path.join("tmp_index_files", save_index_file)

            total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard,
                                                checkpoint=checkpoint,
                                                index_file=index_file,
                                                base_filename=weights_name,
                                                is_master=control_saving,
                                                use_safetensors=use_safetensors)
            if control_saving:
                assert self.dp_rank == 0 and self.tp_rank == 0, "The saving process should have both dp_rank and tp_rank as 0."
                index_file.append_meta_data("total_size", total_size)
                index_file.write_index_file(save_index_file)
            else:
                return

            dist.barrier(self.pp_group)

            # The global master rank integrates the index files and clean the folder.
            if self.pp_rank == 0:
                final_index_file = CheckpointIndexFile(checkpoint)
                final_index_file.append_meta_data("total_size", 0)

                for filename in os.listdir(tmp_index_file_folder):
                    stage_index_file = CheckpointIndexFile.from_file(os.path.join(tmp_index_file_folder, filename))
                    final_index_file.metadata["total_size"] += stage_index_file.metadata["total_size"]
                    for weight, weight_filename in stage_index_file.weight_map.items():
                        final_index_file.append_weight_map(weight, weight_filename)

                final_index_file.write_index_file(final_index_file_path)
                rmtree(tmp_index_file_folder)
                logging.info(f"The model is split into checkpoint shards. "
                             f"You can find where each parameters has been saved in the "
                             f"index located at {final_index_file_path}.")

    def load_sharded_model(self, model: nn.Module, checkpoint_index_file: Path, strict: bool = False):
        """
        Load sharded model with the given path to index file of checkpoint folder.

        Args:
            model (nn.Module): The model to be loaded.
            index_file_path (str): Path to the index file of checkpointing folder.
            strict (bool, optional): For name matching during loading state_dict. Defaults to False.
                                     This argument should be manually set to False since params on same device might be stored in different files.
        """

        # Check whether the checkpoint uses safetensors.
        use_safetensors = False
        if "safetensors" in checkpoint_index_file.name:
            use_safetensors = True

        if use_safetensors and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors` library: `pip install safetensors`.")

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()
        missing_keys = []

        # Load params & buffers to model.
        # Keep a record of loaded files so that file will not be repeatedly loaded.
        strict = False
        for shard_file in checkpoint_files:
            state_dict = load_shard_state_dict(Path(shard_file), use_safetensors)
            load_state_dict_into_model(model, state_dict, missing_keys, strict=strict, load_sub_module=True)
            del state_dict

    def save_sharded_optimizer(self,
                               optimizer: Optimizer,
                               checkpoint: str,
                               gather_dtensor: bool = True,
                               prefix: Optional[str] = None,
                               size_per_shard: int = 1024):
        pass

    def load_sharded_optimizer(self, optimizer: Optimizer, index_file_path: str, prefix: str):
        pass

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool = True):
        # TODO(Baizhou): support this feature after implementing complete state_dict collection
        raise NotImplementedError

    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        # TODO(Baizhou): support this feature after implementing complete state_dict collection
        raise NotImplementedError

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        # TODO(Baizhou): support this feature after implementing complete state_dict collection
        raise NotImplementedError

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        # TODO(Baizhou): support this feature after implementing complete state_dict collection
        raise NotImplementedError

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save lr scheduler to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)


class _StateDictSharder:

    def __init__(self, size_per_shard: int) -> None:
        self.max_shard_size = size_per_shard
        self.current_block = OrderedDict()
        self.current_block_size = 0

    def append(self, name: str, tensor: torch.Tensor) -> Tuple[Optional[OrderedDict], int]:
        tensor_size = calculate_tensor_size(tensor)
        ret_block = None
        ret_block_size = 0

        # before we return the current block and create a new block,
        # we need to ensure that the current block is not empty
        if self.current_block_size + tensor_size > self.max_shard_size and self.current_block_size > 0:
            ret_block = self.current_block
            ret_block_size = self.current_block_size
            self.current_block = OrderedDict()
            self.current_block_size = 0

        self.current_block[name] = tensor
        self.current_block_size += tensor_size
        return ret_block, ret_block_size
