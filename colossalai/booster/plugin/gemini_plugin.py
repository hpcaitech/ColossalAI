import logging
import os
import warnings
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO, GeneralCheckpointIO
from colossalai.checkpoint_io.utils import get_model_base_filenames, get_shard_filename, save_state_dict
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.utils import get_current_device
from colossalai.zero import GeminiDDP, zero_model_wrapper, zero_optim_wrapper
from colossalai.zero.gemini.memory_tracer import MemStats

from .dp_plugin_base import DPPluginBase

__all__ = ['GeminiPlugin']

SUPPORTED_PRECISION = ['fp16', 'bf16']
PRECISION_STR_TO_DTYPE = {'fp16': torch.half, 'bf16': torch.bfloat16}


class GeminiCheckpointIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def save_unsharded_model(self, model: GeminiDDP, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save sharded model to checkpoint but only on master process.
        The model should be unwrapped in self.load_model via ModelWrapper.unwrap.
        As there is communication when getting state dict, this must be called on all processes.
        """
        state_dict = model.state_dict(only_rank_0=True)
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors)

    def load_unsharded_model(self, model: GeminiDDP, checkpoint: str, strict: bool = True):
        """
        Load model from checkpoint with automatic unwrapping.
        The model should be unwrapped in self.load_model via ModelWrapper.unwrap.
        """
        super().load_unsharded_model(model, checkpoint, strict=strict)

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save unsharded optimizer state dict to checkpoint.
        After calling optimizer.state_dict(), the complete optimizer states will be collected on master rank.
        As there is communication when getting state dict, this must be called on all processes.
        The saving process will only be executed by master rank.
        """
        state_dict = optimizer.state_dict()
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors=False)

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str):
        """
        Loading unsharded optimizer from checkpoint file.
        For each process, only loading optimizer states of parameters it controls.
        """
        super().load_unsharded_optimizer(optimizer, checkpoint)

    def save_sharded_model(self,
                           model: GeminiDDP,
                           checkpoint_path: str,
                           gather_dtensor: bool = False,
                           prefix: Optional[str] = None,
                           max_shard_size: int = 1024,
                           use_safetensors: bool = False):
        """
        Save sharded model
        """
        if os.path.isfile(checkpoint_path):
            logging.error(f"Provided path ({checkpoint_path}) should be a directory, not a file")
            return

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        state_dict_shard = model.state_dict_shard(max_shard_size=max_shard_size, only_rank_0=True, dtype=torch.float32)
        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
        total_size = 0
        index_file = CheckpointIndexFile(checkpoint_path)
        for idx, shard_pair in enumerate(state_dict_shard):
            if not self.coordinator.is_master():
                continue
            shard = shard_pair[0]
            shard_file = get_shard_filename(weights_name, idx)
            total_size = total_size + shard_pair[1]
            for key in shard.keys():
                index_file.append_weight_map(key, shard_file)

            checkpoint_file_path = os.path.join(checkpoint_path, shard_file)
            save_state_dict(shard, checkpoint_file_path, use_safetensors)

        index_file.append_meta_data("total_size", total_size)

        # only save the index file on the master rank
        if self.coordinator.is_master():
            index_file.write_index_file(save_index_file)
        logging.info(f"The model is split into checkpoint shards. "
                     f"You can find where each parameters has been saved in the "
                     f"index located at {save_index_file}.")

    def load_sharded_model(self,
                           model: GeminiDDP,
                           checkpoint_index_file: Path,
                           strict: bool = False,
                           use_safetensors: bool = False):
        """
        load shard model, load model from multiple files
        """
        return super().load_sharded_model(model, checkpoint_index_file, strict, use_safetensors, load_sub_module=False)

    def save_sharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool, prefix: str,
                               size_per_shard: int):
        """
        Save sharded optimizer state dict to checkpoint folder.
        As there is communication when getting state dict, this must be called on all processes.
        """
        Path(checkpoint).mkdir(parents=True, exist_ok=True)
        super().save_sharded_optimizer(optimizer, checkpoint, gather_dtensor, prefix, size_per_shard)

    def load_sharded_optimizer(self, optimizer: Optimizer, checkpoint_index_file: Path, prefix: str):
        """
        Loading sharded optimizer from checkpoint folder, with index file given.
        For each process, only loading optimizer states of parameters it controls.
        """
        # TODO(Baizhou): To be implemented.
        pass


class GeminiModel(ModelWrapper):

    def __init__(self, module: nn.Module, gemini_config: dict, verbose: bool = False) -> None:
        super().__init__(module)
        self.module = zero_model_wrapper(module, zero_stage=3, gemini_config=gemini_config, verbose=verbose)

    def unwrap(self):
        # as save/load state dict is coupled with the GeminiDDP, we only return GeminiDDP model
        return self.module


class GeminiOptimizer(OptimizerWrapper):

    def __init__(self,
                 module: GeminiDDP,
                 optimizer: Optimizer,
                 zero_optim_config: dict,
                 optim_kwargs: dict,
                 verbose: bool = False) -> None:
        optimizer = zero_optim_wrapper(module,
                                       optimizer,
                                       optim_config=zero_optim_config,
                                       **optim_kwargs,
                                       verbose=verbose)
        super().__init__(optimizer)

    def backward(self, loss: Tensor, *args, **kwargs):
        self.optim.backward(loss)

    def clip_grad_by_norm(self,
                          max_norm: Union[float, int],
                          norm_type: Union[float, int] = 2,
                          error_if_nonfinite: bool = False,
                          *args,
                          **kwargs) -> Tensor:
        warnings.warn(f'Gemini controls grad clipping by itself, so you should not use clip_grad_by_norm')

    def clip_grad_by_value(self, clip_value: float, *args, **kwargs) -> None:
        raise NotImplementedError('Gemini does not support clip_grad_by_value')


class GeminiPlugin(DPPluginBase):
    """
    Plugin for Gemini.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import GeminiPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = GeminiPlugin()

        >>> train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        device (torch.device): device to place the model.
        placement_policy (str, optional): "cpu", "cuda", "auto". Defaults to "cpu".
        precision (str, optional): precision. Support 'fp16' and 'bf16'. Defaults to 'fp16'.
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
        verbose (bool, optional): verbose mode. Debug info including chunk search result will be printed. Defaults to False.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        placement_policy: str = "cpu",
        precision: str = "fp16",
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
        verbose: bool = False,
    ) -> None:
        super().__init__()
        assert precision in SUPPORTED_PRECISION, f'precision {precision} is not supported'
        self.gemini_config = dict(
            device=(device or get_current_device()),
            placement_policy=placement_policy,
            pin_memory=pin_memory,
            force_outputs_fp32=force_outputs_fp32,
            strict_ddp_mode=strict_ddp_mode,
            search_range_m=search_range_m,
            hidden_dim=hidden_dim,
            min_chunk_size_m=min_chunk_size_m,
            memstats=memstats,
            mixed_precision=PRECISION_STR_TO_DTYPE[precision],
        )
        self.zero_optim_config = dict(gpu_margin_mem_ratio=gpu_margin_mem_ratio,)
        self.optim_kwargs = dict(initial_scale=initial_scale,
                                 growth_factor=growth_factor,
                                 backoff_factor=backoff_factor,
                                 growth_interval=growth_interval,
                                 hysteresis=hysteresis,
                                 min_scale=min_scale,
                                 max_scale=max_scale,
                                 max_norm=max_norm,
                                 norm_type=norm_type)
        self.verbose = verbose

    def support_no_sync(self) -> bool:
        return False

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return SUPPORTED_PRECISION

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ['cuda']

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:

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
            model = GeminiModel(model, self.gemini_config, self.verbose)

        if optimizer is not None and \
                not isinstance(optimizer, OptimizerWrapper):
            optimizer = GeminiOptimizer(model.unwrap(), optimizer, self.zero_optim_config, self.optim_kwargs,
                                        self.verbose)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return GeminiCheckpointIO()

    def no_sync(self, model: nn.Module) -> Iterator[None]:
        raise NotImplementedError
