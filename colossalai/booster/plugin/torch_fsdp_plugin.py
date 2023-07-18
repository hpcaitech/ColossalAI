import warnings
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from torch.distributed import ProcessGroup

if version.parse(torch.__version__) >= version.parse('1.12.0'):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        BackwardPrefetch,
        CPUOffload,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
    )
else:
    raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIO, GeneralCheckpointIO, utils
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper

from .dp_plugin_base import DPPluginBase

__all__ = ['TorchFSDPPlugin']


class TorchFSDPCheckpointIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        checkpoint = utils.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint)

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        checkpoint = utils.load_state_dict(checkpoint)
        fsdp_model = optimizer.unwrap_model()
        sharded_osd = FSDP.scatter_full_optim_state_dict(checkpoint, fsdp_model)
        optimizer.load_state_dict(sharded_osd)

    def save_unsharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        """
        Save model to checkpoint but only on master process.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            full_model_state = model.state_dict()
        utils.save_state_dict(full_model_state, checkpoint_file_path=checkpoint, use_safetensors=use_safetensors)

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool):
        """
        Save optimizer to checkpoint but only on master process.
        """
        assert isinstance(optimizer, FSDPOptimizerWrapper)
        fsdp_model = optimizer.unwrap_model()
        full_optimizer_state = FSDP.full_optim_state_dict(fsdp_model, optim=optimizer, rank0_only=True)
        utils.save_state_dict(full_optimizer_state, checkpoint_file_path=checkpoint, use_safetensors=False)

    def save_sharded_model(self, model: nn.Module, checkpoint: str, gather_dtensor: bool, prefix: Optional[str],
                           size_per_shard: int, use_safetensors: bool):
        """
        Save model to checkpoint but only on master process.
        """
        raise NotImplementedError("Sharded model checkpoint is not supported yet.")

    def load_sharded_model(self,
                           model: nn.Module,
                           checkpoint_index_file: Path,
                           strict: bool = False,
                           use_safetensors: bool = False,
                           load_sub_module: bool = True):
        """
        Load model to checkpoint but only on master process.
        """
        raise NotImplementedError("Sharded model checkpoint is not supported yet.")

    def save_sharded_optimizer(self, optimizer: Optimizer, checkpoint: str, gather_dtensor: bool, prefix: str,
                               size_per_shard: int):
        """
        Save optimizer to checkpoint but only on master process.
        """
        raise NotImplementedError("Sharded optimizer checkpoint is not supported yet.")

    def load_sharded_optimizer(self, optimizer: Optimizer, index_file_path: str, size_per_shard: int):
        """
        Load optimizer to checkpoint but only on master process.
        """
        raise NotImplementedError("Sharded optimizer checkpoint is not supported yet.")

    def save_lr_scheduler(self, lr_scheduler: LRScheduler, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        if self.coordinator.is_master():
            super().save_lr_scheduler(lr_scheduler, checkpoint)


class TorchFSDPModel(ModelWrapper):

    def __init__(self, module: nn.Module, *args, **kwargs) -> None:
        super().__init__(module)
        self.module = FSDP(module, *args, **kwargs)

    def unwrap(self):
        return self.module


class FSDPOptimizerWrapper(OptimizerWrapper):

    def __init__(self, optimizer: Optimizer, model: nn.Module):
        self.model = model
        super().__init__(optimizer)

    def unwrap_model(self) -> nn.Module:
        return self.model


class TorchFSDPPlugin(DPPluginBase):
    """
    Plugin for PyTorch FSDP.

    Example:
        >>> from colossalai.booster import Booster
        >>> from colossalai.booster.plugin import TorchFSDPPlugin
        >>>
        >>> model, train_dataset, optimizer, criterion = ...
        >>> plugin = TorchFSDPPlugin()

        >>> train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)
        >>> booster = Booster(plugin=plugin)
        >>> model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)

    Args:
        See https://pytorch.org/docs/stable/fsdp.html for details.
    """

    if version.parse(torch.__version__) >= version.parse('1.12.0'):

        def __init__(
            self,
            process_group: Optional[ProcessGroup] = None,
            sharding_strategy: Optional[ShardingStrategy] = None,
            cpu_offload: Optional[CPUOffload] = None,
            auto_wrap_policy: Optional[Callable] = None,
            backward_prefetch: Optional[BackwardPrefetch] = None,
            mixed_precision: Optional[MixedPrecision] = None,
            ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
            param_init_fn: Optional[Callable[[nn.Module], None]] = None,
            sync_module_states: bool = False,
        ):
            super().__init__()
            self.fsdp_kwargs = dict(process_group=process_group,
                                    sharding_strategy=sharding_strategy,
                                    cpu_offload=cpu_offload,
                                    auto_wrap_policy=auto_wrap_policy,
                                    backward_prefetch=backward_prefetch,
                                    mixed_precision=mixed_precision,
                                    ignored_modules=ignored_modules,
                                    param_init_fn=param_init_fn,
                                    sync_module_states=sync_module_states)
    else:
        raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")

    def support_no_sync(self) -> bool:
        False

    def no_sync(self, model: nn.Module) -> Iterator[None]:
        raise NotImplementedError("Torch fsdp no_sync func not supported yet.")

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return ['fp16', 'bf16']

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

        # wrap the model with PyTorch FSDP
        fsdp_model = TorchFSDPModel(model, device_id=torch.cuda.current_device(), **self.fsdp_kwargs)

        if optimizer is not None:
            if len(optimizer.param_groups) > 1:
                warnings.warn(
                    'TorchFSDPPlugin does not support optimizer that use multi param groups. The results may not be as expected if used.'
                )
            optimizer.__init__(fsdp_model.parameters(), **optimizer.defaults)

            if not isinstance(optimizer, FSDPOptimizerWrapper):
                optimizer = FSDPOptimizerWrapper(optimizer, fsdp_model)

        return fsdp_model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return TorchFSDPCheckpointIO()
