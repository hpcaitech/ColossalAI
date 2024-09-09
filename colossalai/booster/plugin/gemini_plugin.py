import gc
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.distributed_c10d import _get_default_group
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from colossalai.accelerator import get_accelerator
from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO, GeneralCheckpointIO
from colossalai.checkpoint_io.utils import (
    get_model_base_filenames,
    get_optimizer_base_filenames,
    load_shard_state_dict,
    save_config_file,
    save_state_dict,
    save_state_dict_shards,
)
from colossalai.cluster import DistCoordinator, ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.zero import GeminiDDP, GeminiOptimizer
from colossalai.zero.gemini.memory_tracer import MemStats

from .dp_plugin_base import DPPluginBase

__all__ = ["GeminiPlugin"]

SUPPORTED_PRECISION = ["fp16", "bf16"]
PRECISION_STR_TO_DTYPE = {"fp16": torch.half, "bf16": torch.bfloat16}

ZERO_AXIS, DP_AXIS, TP_AXIS = 0, 1, 2


def get_param_info(optim: Optimizer):
    # Get a backup of necessary information of parameters for future use, which includes:
    # 1. A mapping from integer param_id to param32 shape.
    if optim is None:
        return {}
    param_info = {"id2shape": {}}

    start_index = 0
    for group in optim.param_groups:
        for param_id, param in enumerate(group["params"], start_index):
            original_shape = param.shape if isinstance(param, torch.Tensor) else None
            param_info["id2shape"][param_id] = original_shape

        start_index += len(group["params"])

    return param_info


class GeminiCheckpointIO(GeneralCheckpointIO):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()
        self.logger = get_dist_logger()

    def save_unsharded_model(self, model: GeminiDDP, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save sharded model to checkpoint but only on master process.
        The model should be unwrapped in self.load_model via ModelWrapper.unwrap.
        As there is communication when getting state dict, model.state_dict() must be called on all processes.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before saving!"
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors)

    def load_unsharded_model(self, model: GeminiDDP, checkpoint: str, strict: bool = True):
        """
        Load model from checkpoint with automatic unwrapping.
        The model should be unwrapped in self.load_model via ModelWrapper.unwrap.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before loading!"
        super().load_unsharded_model(model, checkpoint, strict=strict)

    def save_unsharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save unsharded optimizer state dict to checkpoint.
        After calling optimizer.state_dict(), the complete optimizer states will be collected on master rank.
        As there is communication when getting state dict, optimizer.state_dict() must be called on all processes.
        The saving process will only be executed by master rank.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before saving!"
        state_dict = optimizer.state_dict()
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors=False)

    def load_unsharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint: str):
        """
        Loading unsharded optimizer from checkpoint file.
        For each process, only loading optimizer states of parameters it controls.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before loading!"
        super().load_unsharded_optimizer(optimizer, checkpoint)

    def save_sharded_model(
        self,
        model: GeminiDDP,
        checkpoint_path: str,
        gather_dtensor: bool = False,
        prefix: Optional[str] = None,
        max_shard_size: int = 1024,
        use_safetensors: bool = False,
    ):
        """
        Save sharded model.
        As there is communication when getting state dict, model.state_dict() must be called on all processes.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before saving!"
        if os.path.isfile(checkpoint_path):
            self.logger.error(f"Provided path ({checkpoint_path}) should be a directory, not a file", ranks=[0])
            return

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        state_dict_shard = model.state_dict_shard(max_shard_size=max_shard_size, only_rank_0=True)
        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
        index_file = CheckpointIndexFile(checkpoint_path)

        # Save shards of optimizer states.
        is_master = self.coordinator.is_master()
        total_size = save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint_path,
            index_file=index_file,
            base_filename=weights_name,
            is_master=is_master,
            use_safetensors=use_safetensors,
        )

        # only save the index file on the master rank
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            save_config_file(model.unwrap(), checkpoint_path)
            self.logger.info(
                f"The model is split into checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {save_index_file}.",
                ranks=[0],
            )

    def load_sharded_model(
        self, model: GeminiDDP, checkpoint_index_file: Path, strict: bool = False, use_safetensors: bool = False
    ):
        """
        Load shard model, load model from multiple files.
        """
        assert isinstance(model, GeminiDDP), "Please boost the model before loading!"
        return super().load_sharded_model(model, checkpoint_index_file, strict, use_safetensors, load_sub_module=False)

    def save_sharded_optimizer(
        self, optimizer: GeminiOptimizer, checkpoint: Path, gather_dtensor: bool, prefix: str, size_per_shard: int
    ):
        """
        Save sharded optimizer state dict to checkpoint folder.
        As there is communication when getting state dict, this must be called on all processes.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before saving!"

        if os.path.isfile(checkpoint):
            self.logger.error(f"Provided path ({checkpoint}) should be a directory, not a file", ranks=[0])
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # Preparing file paths and index file.
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix)
        index_file = CheckpointIndexFile(checkpoint)
        index_file.append_meta_data("param_groups", param_group_file)

        # Store the information of param groups to param_group_file.
        if self.coordinator.is_master():
            group_file_path = os.path.join(checkpoint, param_group_file)
            param_groups = optimizer.get_param_groups_for_saving()
            torch.save(param_groups, group_file_path)

        # States are broken into shards within max_shard_size.
        state_dict_shard = optimizer.state_shard(prefix=prefix, max_shard_size=size_per_shard, only_rank_0=True)

        # Save shards of optimizer states.
        total_size = save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint,
            index_file=index_file,
            base_filename=states_name,
            is_master=self.coordinator.is_master(),
            use_safetensors=False,
        )

        # Wrap up index file. Only save it on master rank.
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            self.logger.info(
                f"The optimizer is going to be split to checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {save_index_file}.",
                ranks=[0],
            )

    def load_sharded_optimizer(self, optimizer: GeminiOptimizer, checkpoint_index_file: Path, prefix: str):
        """
        Loading sharded optimizer from checkpoint folder, with index file given.
        For each process, only loading optimizer states of parameters it controls.
        """
        assert isinstance(optimizer, GeminiOptimizer), "Please boost the optimizer before loading!"
        if not os.path.isfile(checkpoint_index_file):
            self.logger.error(f"Provided path ({checkpoint_index_file}) should be a file", ranks=[0])

        assert isinstance(optimizer, GeminiOptimizer)

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)

        # Load param_groups.
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(
                f"Invalid index file path {checkpoint_index_file} for an optimizer. \
                               Lacking param group file under current directory."
            )
        saved_param_groups = torch.load(param_group_path)
        optimizer.load_param_groups(saved_param_groups)

        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()

        # Load optimizer states from shard files under checkpoint path.
        # For each file, only load the states managed by current process.
        for shard_file in checkpoint_files:
            state_dict_shard = load_shard_state_dict(Path(shard_file), use_safetensors=False)
            optimizer.load_param_states(state_dict_shard)
            del state_dict_shard
            gc.collect()

        optimizer.optimizer_loading_epilogue()

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)


class GeminiPlugin(DPPluginBase):
    """
    Plugin for Gemini.

    ```python
    from colossalai.booster import Booster
    from colossalai.booster.plugin import GeminiPlugin

    model, train_dataset, optimizer, criterion = ...
    plugin = GeminiPlugin()

    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
    booster = Booster(plugin=plugin)
    model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)
    ```

    Args:
        chunk_config_dict (dict, optional): chunk configuration dictionary.
        chunk_init_device (torch.device, optional): device to initialize the chunk.
        placement_policy (str, optional): "static" and "auto". Defaults to "static".
        enable_gradient_accumulation (bool, optional): Whether to enable gradient accumulation. When set to True, gradient will be stored after doing backward pass. Defaults to False.
        shard_param_frac (float, optional): fraction of parameters to be sharded. Only for "static" placement.
            If `shard_param_frac` is 1.0, it's equal to zero-3. If `shard_param_frac` is 0.0, it's equal to zero-2. Defaults to 1.0.
        offload_optim_frac (float, optional): fraction of optimizer states to be offloaded. Only for "static" placement.
            If `shard_param_frac` is 1.0 and `offload_optim_frac` is 0.0, it's equal to old "cuda" placement. Defaults to 0.0.
        offload_param_frac (float, optional): fraction of parameters to be offloaded. Only for "static" placement.
            For efficiency, this argument is useful only when `shard_param_frac` is 1.0 and `offload_optim_frac` is 1.0.
            If `shard_param_frac` is 1.0, `offload_optim_frac` is 1.0 and `offload_param_frac` is 1.0, it's equal to old "cpu" placement.
            When using static placement, we recommend users to tune `shard_param_frac` first and then `offload_optim_frac`.
            Defaults to 0.0.
        warmup_non_model_data_ratio (float, optional): ratio of expected non-model data memory during warmup. Only for "auto" placement. Defaults to 0.8.
        steady_cuda_cap_ratio (float, optional): ratio of allowed cuda capacity for model data during steady state. Only for "auto" placement. Defaults to 0.9.
        precision (str, optional): precision. Support 'fp16' and 'bf16'. Defaults to 'fp16'.
        master_weights (bool, optional): Whether to keep fp32 master parameter weights in optimizer. Defaults to True.
        pin_memory (bool, optional): use pin memory on CPU. Defaults to False.
        force_outputs_fp32 (bool, optional): force outputs are fp32. Defaults to False.
        strict_ddp_mode (bool, optional): use strict ddp mode (only use dp without other parallelism). Defaults to False.
        search_range_m (int, optional): chunk size searching range divided by 2^20. Defaults to 32.
        hidden_dim (int, optional): the hidden dimension of DNN.
            Users can provide this argument to speed up searching.
            If users do not know this argument before training, it is ok. We will use a default value 1024.
        min_chunk_size_m (float, optional): the minimum chunk size divided by 2^20.
            If the aggregate size of parameters is still smaller than the minimum chunk size,
            all parameters will be compacted into one small chunk.
        memstats (MemStats, optional) the memory statistics collector by a runtime memory tracer.
        gpu_margin_mem_ratio (float, optional): The ratio of GPU remaining memory (after the first forward-backward)
            which will be used when using hybrid CPU optimizer.
            This argument is meaningless when `placement_policy` of `GeminiManager` is not "auto".
            Defaults to 0.0.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**16.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        max_norm (float, optional): max_norm used for `clip_grad_norm`. You should notice that you shall not do
            clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.
        norm_type (float, optional): norm_type used for `clip_grad_norm`.
        tp_size (int, optional): If 'tp_size' is set to be greater than 1, it means using tensor parallelism strategy, which is implemented in Shardformer, 'tp_size' determines the size of the tensor parallel process group. Default to 1.
        extra_dp_size (int, optional): If 'extra_dp_size' is set to be greater than 1, it means creating another group to run with a ddp-like strategy. Default to 1.
        enable_all_optimization (bool, optional): Whether to switch on all the optimizations supported by Shardformer.
                                                    Currently all the optimization methods include fused normalization, flash attention and JIT.
                                                    Defaults to False.
        enable_fused_normalization (bool, optional): Whether to switch on fused normalization in Shardformer. Defaults to False.
        enable_flash_attention (bool, optional): Whether to switch on flash attention in Shardformer. Defaults to False.
        enable_jit_fused (bool, optional): Whether to switch on JIT in Shardformer. Default to False.
        enable_sequence_parallelism (bool): Whether to turn on sequence parallelism in Shardformer. Defaults to False.
        enable_sequence_overlap (bool): Whether to turn on sequence overlap in Shardformer. Defaults to False.
        use_fp8 (bool, optional): Whether to enable fp8 mixed precision training. Defaults to False.
        verbose (bool, optional): verbose mode. Debug info including chunk search result will be printed. Defaults to False.
        fp8_communication (bool, optional): Whether to enable fp8 communication. Defaults to False.
    """

    def __init__(
        self,
        chunk_config_dict: Optional[dict] = None,
        chunk_init_device: Optional[torch.device] = None,
        placement_policy: str = "static",
        enable_gradient_accumulation: bool = False,
        max_prefetch: int = 0,
        shard_param_frac: float = 1.0,  # only for static placement
        offload_optim_frac: float = 0.0,  # only for static placement
        offload_param_frac: float = 0.0,  # only for static placement
        warmup_non_model_data_ratio: float = 0.8,  # only for auto placement
        steady_cuda_cap_ratio: float = 0.9,  # only for auto placement
        precision: str = "fp16",
        master_weights: bool = True,
        pin_memory: bool = False,
        force_outputs_fp32: bool = False,
        strict_ddp_mode: bool = False,
        search_range_m: int = 32,
        hidden_dim: Optional[int] = None,
        min_chunk_size_m: float = 32,
        memstats: Optional[MemStats] = None,
        gpu_margin_mem_ratio: float = 0.0,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
        tp_size: int = 1,
        extra_dp_size: int = 1,
        enable_all_optimization: bool = False,
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_sequence_parallelism: bool = False,
        enable_jit_fused: bool = False,
        enable_sequence_overlap: bool = False,
        enable_async_reduce: bool = True,
        use_fp8: bool = False,
        verbose: bool = False,
        fp8_communication: bool = False,
    ) -> None:
        super().__init__()
        assert precision in SUPPORTED_PRECISION, f"precision {precision} is not supported"
        if get_accelerator().name == "npu":
            assert placement_policy == "static", "NPU only supports static placement policy"

        self.logger = get_dist_logger()
        if enable_async_reduce and not pin_memory:
            self.logger.warning(
                f"enable_async_reduce sets pin_memory=True to achieve best performance, which is not implicitly set.",
                ranks=[0],
            )
            pin_memory = True
        self.gemini_config = dict(
            chunk_config_dict=chunk_config_dict,
            chunk_init_device=(chunk_init_device or get_accelerator().get_current_device()),
            placement_policy=placement_policy,
            enable_gradient_accumulation=enable_gradient_accumulation,
            shard_param_frac=shard_param_frac,
            offload_optim_frac=offload_optim_frac,
            offload_param_frac=offload_param_frac,
            warmup_non_model_data_ratio=warmup_non_model_data_ratio,
            steady_cuda_cap_ratio=steady_cuda_cap_ratio,
            pin_memory=pin_memory,
            force_outputs_fp32=force_outputs_fp32,
            strict_ddp_mode=strict_ddp_mode,
            search_range_m=search_range_m,
            hidden_dim=hidden_dim,
            min_chunk_size_m=min_chunk_size_m,
            memstats=memstats,
            mixed_precision=PRECISION_STR_TO_DTYPE[precision],
            master_weights=master_weights,
            max_prefetch=max_prefetch,
            enable_async_reduce=enable_async_reduce,
            fp8_communication=fp8_communication,
            use_fp8=use_fp8,
        )
        self.zero_optim_config = dict(
            gpu_margin_mem_ratio=gpu_margin_mem_ratio,
        )
        self.optim_kwargs = dict(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        self.enable_tensor_parallelism = tp_size > 1
        self.enable_all_optimization = enable_all_optimization
        self.enable_fused_normalization = enable_fused_normalization
        self.enable_flash_attention = enable_flash_attention
        self.enable_sequence_parallelism = enable_sequence_parallelism if self.enable_tensor_parallelism else False
        self.enable_jit_fused = enable_jit_fused
        self.enable_sequence_overlap = enable_sequence_overlap
        self.verbose = verbose

        self.tp_size = tp_size
        self.extra_dp_size = extra_dp_size
        world_size = dist.get_world_size()
        self.zero_size = world_size // (self.tp_size * self.extra_dp_size)
        assert (
            world_size == (self.tp_size * self.extra_dp_size) * self.zero_size
        ), f"The global group size can't be evenly divided by the subgroup size."

        self.pg_mesh = ProcessGroupMesh(self.zero_size, self.extra_dp_size, self.tp_size)
        self.zero_group = (
            self.pg_mesh.get_group_along_axis(ZERO_AXIS) if self.zero_size < world_size else _get_default_group()
        )
        self.extra_dp_group = self.pg_mesh.get_group_along_axis(DP_AXIS) if self.extra_dp_size > 1 else None
        self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS) if self.tp_size > 1 else None
        self.dp_size = self.zero_size * self.extra_dp_size

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=self.tp_group,
            enable_tensor_parallelism=self.enable_tensor_parallelism,
            enable_all_optimization=self.enable_all_optimization,
            enable_fused_normalization=self.enable_fused_normalization,
            enable_flash_attention=self.enable_flash_attention,
            enable_jit_fused=self.enable_jit_fused,
            enable_sequence_parallelism=self.enable_sequence_parallelism,
            enable_sequence_overlap=self.enable_sequence_overlap,
        )

    def __del__(self):
        """Destroy the process groups in ProcessGroupMesh"""
        self.pg_mesh.destroy_mesh_process_groups()

    def support_no_sync(self) -> bool:
        return False

    def support_lora(self) -> bool:
        return False

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return SUPPORTED_PRECISION

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ["cuda", "npu"]

    def prepare_dataloader(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=1024,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        distributed_sampler_cls=None,
        **kwargs,
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
        zero_world_size = self.pg_mesh.size(ZERO_AXIS)
        extra_dp_world_size = self.pg_mesh.size(DP_AXIS)
        zero_rank = self.pg_mesh.coordinate(ZERO_AXIS)
        extra_dp_rank = self.pg_mesh.coordinate(DP_AXIS)
        distributed_sampler_cls = distributed_sampler_cls or DistributedSampler
        sampler = distributed_sampler_cls(
            dataset,
            num_replicas=zero_world_size * extra_dp_world_size,
            rank=zero_rank * extra_dp_world_size + extra_dp_rank,
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

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        params_info = get_param_info(optimizer)
        if not isinstance(model, ModelWrapper):
            # convert model to sync bn
            # FIXME(ver217): gemini does not support sync bn
            # In torch/nn/modules/_functions.py, line 22, ``mean, invstd = torch.batch_norm_stats(input, eps)`` will get fp32 mean and invstd even though the input is fp16.
            # This inconsistency of dtype will cause the error.
            # We have two possible solutions:
            # 1. keep batch norm always in fp32. This is hard for gemini, as it use chunks.
            # 2. patch sync bn or write a new on. This is relatively easy, but we need to test it.
            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model, None)

            # wrap the model with Gemini
            if self.enable_tensor_parallelism:
                shardformer = ShardFormer(self.shard_config)
                model, _ = shardformer.optimize(model)

            model = GeminiDDP(
                model,
                **self.gemini_config,
                zero_group=self.zero_group,
                extra_dp_group=self.extra_dp_group,
                verbose=self.verbose,
            )

        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            optimizer = GeminiOptimizer(
                optimizer,
                model,
                **self.zero_optim_config,
                **self.optim_kwargs,
                tp_group=self.tp_group,
                params_info=params_info,
                verbose=self.verbose,
            )

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return GeminiCheckpointIO()

    def no_sync(self, model: nn.Module, optimizer: OptimizerWrapper) -> Iterator[None]:
        raise NotImplementedError

    def enable_lora(
        self, model: nn.Module, pretrained_dir: Optional[str] = None, lora_config: Optional[Dict] = None
    ) -> nn.Module:
        raise NotImplementedError
