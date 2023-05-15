from typing import Callable, Iterable, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from torch.distributed import ProcessGroup

if version.parse(torch.__version__) >= version.parse('1.12.0') and version.parse(
        torch.__version__) < version.parse('2.0.0'):
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        BackwardPrefetch,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
    )
elif version.parse(torch.__version__) >= version.parse('2.0.0'):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp._init_utils import ProcessGroupType
    from torch.distributed.fsdp.api import (
        BackwardPrefetch,
        CPUOffload,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import _FSDPPolicy
else:
    raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from colossalai.checkpoint_io import CheckpointIO, GeneralCheckpointIO
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper

from .dp_plugin_base import DPPluginBase

__all__ = ['TorchFSDPPlugin']


class TorchFSDPCheckpointIO(GeneralCheckpointIO):

    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()

    def __set_model_optim_state(
        self,
        model,
        state_dict_type,
        state_dict_config,
        optim_state_dict_config,
    ):
        return FSDP.set_state_dict_type(model, state_dict_type, state_dict_config, optim_state_dict_config)

    def load_sharded_model(self, model: nn.Module, checkpoint: str):

        # TODO(jishaomin): implement this method as it can be supported by Huggingface model
        raise NotImplementedError("Torch FSDP sharded model checkpoint is not supported yet.")

    def load_sharded_optimizer(self, model: nn.Module, optimizer: Optimizer, checkpoint: str):

        # TODO(jishaomin): implement this method as it can be supported by Huggingface model
        raise NotImplementedError("Torch FSDP sharded model checkpoint is not supported yet.")

    def save_sharded_model(self, model: nn.Module, checkpoint: str):

        # TODO(jishaomin): implement this method as it can be supported by Huggingface model
        raise NotImplementedError("Torch FSDP sharded model checkpoint is not supported yet.")

    def save_sharded_optimizer(self, model: nn.Module, optimizer: Optimizer, checkpoint: str):

        # TODO(jishaomin): implement this method as it can be supported by Huggingface model
        raise NotImplementedError("Torch FSDP sharded model checkpoint is not supported yet.")

    def load_unsharded_model(self, model: nn.Module, checkpoint: str):
        """
        Load model from checkpoint with automatic unwrapping.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap

        if version.parse(torch.__version__) >= version.parse('1.12.0') and version.parse(
                torch.__version__) < version.parse('2.0.0'):
            full_state_dict = self.load_state_dict(checkpoint)
        elif version.parse(torch.__version__) >= version.parse('2.0.0'):
            full_state_dict = self.load_state_dict(checkpoint)
            self.__set_model_optim_state(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True))
            full_state_dict = model.state_dict()
        else:
            raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")

        model.load_state_dict(full_state_dict)

    def load_unsharded_optimizer(self, model: nn.Module, optim: Optimizer, checkpoint: str):
        """
        Load Optimizer from checkpoint with automatic unwrapping.
        """

        if version.parse(torch.__version__) >= version.parse('1.12.0') and version.parse(
                torch.__version__) < version.parse('2.0.0'):
            optim_full_state_dict = self.load_state_dict(checkpoint)
        elif version.parse(torch.__version__) >= version.parse('2.0.0'):
            optim_full_state_dict = self.load_state_dict(checkpoint)
            FSDP.full_optim_state_dict_to_load(optim_full_state_dict, model, optim)
        else:
            raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")

        optim.load_state_dict(optim_full_state_dict)

    def save_unsharded_model(self, model: nn.Module, checkpoint: str):
        """
        Save model to checkpoint but only on master process.
        """
        # the model should be unwrapped in self.load_model via ModelWrapper.unwrap

        if version.parse(torch.__version__) >= version.parse('1.12.0') and version.parse(
                torch.__version__) < version.parse('2.0.0'):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                model_state_dict = model.state_dict()
        elif version.parse(torch.__version__) >= version.parse('2.0.0'):
            self.__set_model_optim_state(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True))
            model_state_dict = model.state_dict()
        else:
            raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")
        self.save_checkpoint(model_state_dict, checkpoint)

    def save_unsharded_optimizer(self, model: nn.Module, optimizer: Optimizer, checkpoint: str):
        """
        Save optimizer to checkpoint but only on master process.
        """

        if version.parse(torch.__version__) >= version.parse('1.12.0') and version.parse(
                torch.__version__) < version.parse('2.0.0'):
            optim_state_dict = FSDP.full_optim_state_dict(model=model, optim=optimizer)
        elif version.parse(torch.__version__) >= version.parse('2.0.0'):
            self.__set_model_optim_state(model, StateDictType.FULL_STATE_DICT,
                                         FullOptimStateDictConfig(rank0_only=True))
            optim_state_dict = FSDP.optim_state_dict(model, optimizer)
        else:
            raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")
        self.save_checkpoint(optim_state_dict, checkpoint)


class TorchFSDPModel(ModelWrapper):

    def __init__(self, module: nn.Module, *args, **kwargs) -> None:
        super().__init__(module)
        self.module = FSDP(module, *args, **kwargs)

    def unwrap(self):
        return self.module.module


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

    if version.parse(torch.__version__) >= version.parse('1.12.0') and version.parse(
            torch.__version__) < version.parse('2.0.0'):

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
            device_id: Optional[Union[int, torch.device]] = None,
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
                                    device_id=device_id,
                                    sync_module_states=sync_module_states)
    elif version.parse(torch.__version__) >= version.parse('2.0.0'):

        def __init__(
            self,
            process_group: ProcessGroupType = None,
            sharding_strategy: Optional[ShardingStrategy] = None,
            cpu_offload: Optional[CPUOffload] = None,
            auto_wrap_policy: Optional[Union[Callable, _FSDPPolicy]] = None,
            backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
            mixed_precision: Optional[MixedPrecision] = None,
            ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
            param_init_fn: Optional[Callable[[nn.Module], None]] = None,
            device_id: Optional[Union[int, torch.device]] = None,
            sync_module_states: bool = False,
            forward_prefetch: bool = False,
            limit_all_gathers: bool = False,
            use_orig_params: bool = False,
            ignored_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
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
                                    device_id=device_id,
                                    sync_module_states=sync_module_states,
                                    forward_prefetch=forward_prefetch,
                                    limit_all_gathers=limit_all_gathers,
                                    use_orig_params=use_orig_params,
                                    ignored_parameters=ignored_parameters)
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
        optimizer: Optimizer,
        criterion: Callable = None,
        dataloader: DataLoader = None,
        lr_scheduler: LRScheduler = None,
    ) -> Tuple[Union[nn.Module, OptimizerWrapper, LRScheduler, DataLoader]]:

        model = model.cuda()
        # wrap the model with PyTorch FSDP
        model = TorchFSDPModel(model, **self.fsdp_kwargs)

        if not isinstance(optimizer, OptimizerWrapper):
            optimizer = OptimizerWrapper(optimizer)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return TorchFSDPCheckpointIO()
