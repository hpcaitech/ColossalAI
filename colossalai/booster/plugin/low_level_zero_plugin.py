import logging
import os
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Callable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader

from colossalai.accelerator import get_accelerator
from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO
from colossalai.checkpoint_io.utils import (
    get_optimizer_base_filenames,
    get_shard_filename,
    load_param_groups_into_optimizer,
    load_shard_state_dict,
    load_states_into_optimizer,
    save_param_groups,
    save_state_dict,
    sharded_optimizer_loading_epilogue,
)
from colossalai.interface import AMPModelMixin, ModelWrapper, OptimizerWrapper
from colossalai.interface.optimizer import DistributedOptimizer
from colossalai.zero import LowLevelZeroOptimizer

from .dp_plugin_base import DPPluginBase
from .torch_ddp_plugin import TorchDDPCheckpointIO

__all__ = ["LowLevelZeroPlugin"]


def _convert_floating_point(x, dtype: torch.dtype = torch.float16):
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype)
    return x


SUPPORTED_PRECISION = ["fp16", "bf16", "fp32"]


class LowLevelZeroModel(ModelWrapper, AMPModelMixin):
    def __init__(self, module: nn.Module, precision: str) -> None:
        super().__init__(module)
        self.dtype = None
        if precision == "fp16":
            self.dtype = torch.float16
        elif precision == "bf16":
            self.dtype = torch.bfloat16
        if self.dtype is not None:
            module = module.to(self.dtype)
        module = module.to(get_accelerator().get_current_device())
        self.module = module
        self.convert_fn = None
        if self.dtype is not None:
            self.convert_fn = partial(_convert_floating_point, dtype=self.dtype)

    def forward(self, *args, **kwargs):
        if self.convert_fn is not None:
            args = tree_map(self.convert_fn, args)
            kwargs = tree_map(self.convert_fn, kwargs)
        return super().forward(*args, **kwargs)


class LowLevelZeroCheckpointIO(TorchDDPCheckpointIO):
    def save_unsharded_optimizer(self, optimizer: OptimizerWrapper, checkpoint: str, gather_dtensor: bool = False):
        """Save optimizer to checkpoint but only on master process.

        Args:
            optimizer (OptimizerWrapper): Optimizer to save state_dict
            checkpoint (str): Path to save checkpoint
            gather_dtensor (bool): Whether to gather_dtensor, not used
        """
        assert isinstance(optimizer, LowLevelZeroOptimizer), "Please boost the optimizer before saving!"
        # the `state_dict` in LowLevelZeroOptimizer has communication
        # if only the master rank collect state_dict and save,
        # the communication on each rank would not match
        state_dict = optimizer.state_dict()
        if self.coordinator.is_master():
            save_state_dict(state_dict, checkpoint, use_safetensors=False)

    def save_sharded_optimizer(
        self,
        optimizer: OptimizerWrapper,
        checkpoint: str,
        gather_dtensor: bool = False,
        prefix: str = None,
        size_per_shard: int = 1024,
    ):
        """
        Save sharded Zero-optimizer checkpoint under the given checkpointing path.
        The following files will be created under the path:
        - An index file (pytorch_optim.bin.index.json) containing a map between optimizer states and file names
        - A group file (pytorch_optim_group.bin) recording information of param_groups
        - Multiple files (pytorch_optim-000XX.bin) that store state tensors of optimizer in a sharding way

        Args:
            optimizer (OptimizerWrapper): Optimizer to save sharded state_dict
            checkpoint (str): Path to save optimizer state_dict
            gather_dtensor (bool): Whether to gather_dtensor, not used
            prefix (str): Perfix of file to save
            size_per_shard (int): Max file size of each file that store state tensors
        """
        assert isinstance(optimizer, LowLevelZeroOptimizer), "Please boost the optimizer before saving!"
        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # state_dict only provide only 'param_groups'
        state_dict = optimizer.optim.state_dict()
        # state shard would be handled by the low-level zero optimizer
        sharded_state = optimizer.state_dict_shard(max_shard_size=size_per_shard)

        # Preparing file paths and index file.
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix)
        index_file = CheckpointIndexFile(checkpoint)
        index_file.append_meta_data("param_groups", param_group_file)

        # Store the information of param groups to param_group_file.
        if self.coordinator.is_master():
            group_file_path = os.path.join(checkpoint, param_group_file)
            save_param_groups(state_dict, group_file_path)

        # Save shards of optimizer states.
        total_size = 0
        for idx, shard_pair in enumerate(sharded_state):
            shard, current_size = shard_pair
            shard_file = get_shard_filename(states_name, idx)
            total_size = total_size + current_size
            for param_id in shard.keys():
                index_file.append_weight_map(str(param_id), shard_file)

            checkpoint_file_path = os.path.join(checkpoint, shard_file)
            if self.coordinator.is_master():
                save_state_dict(shard, checkpoint_file_path, use_safetensors=False)

        # Wrap up index file.
        index_file.append_meta_data("total_size", total_size)
        if self.coordinator.is_master():
            index_file.write_index_file(save_index_file)
        logging.info(
            f"The optimizer is going to be split to checkpoint shards. "
            f"You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    def load_sharded_optimizer(self, optimizer: OptimizerWrapper, index_file_path: str, prefix: str):
        """Load sharded optimizer with the given path to index file.

        Args:
            optimizer (OptimizerWrapper): Optimizer to load state_dict
            index_file_path (str): Path to the index file
            prefix (str): Not used.
        """
        assert isinstance(optimizer, LowLevelZeroOptimizer), "Please boost the optimizer before Loading!"
        optimizer = optimizer.unwrap()

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(index_file_path)

        # Load param_groups
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(
                f"Invalid index file path {index_file_path} for an optimizer. \
                               Lacking param group file under current directory."
            )
        id_map = load_param_groups_into_optimizer(optimizer, param_group_path)

        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()

        for shard_file in checkpoint_files:
            state_dict = load_shard_state_dict(Path(shard_file), use_safetensors=False)
            # shard state dict
            for param_idx, state in state_dict.items():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor) and k != "step":
                        padding_size = (
                            self.coordinator.world_size - v.numel() % self.coordinator.world_size
                        ) % self.coordinator.world_size
                        with torch.no_grad():
                            v = v.flatten()
                            if padding_size > 0:
                                v = torch.nn.functional.pad(v, [0, padding_size])
                            v_list = v.split(v.numel() // self.coordinator.world_size)
                            state_dict[param_idx][k] = v_list[self.coordinator.rank].detach().clone()
            load_states_into_optimizer(optimizer, state_dict, id_map)
        sharded_optimizer_loading_epilogue(optimizer)

    def load_unsharded_model(self, model: ModelWrapper, checkpoint: str, strict: bool = True):
        assert isinstance(model, LowLevelZeroModel), "Please boost the model before loading!"
        super().load_unsharded_model(model, checkpoint, strict)
        model.update_master_params()

    def load_sharded_model(
        self,
        model: ModelWrapper,
        checkpoint_index_file: Path,
        strict: bool = False,
        use_safetensors: bool = False,
        load_sub_module: bool = True,
    ):
        assert isinstance(model, LowLevelZeroModel), "Please boost the model before loading!"
        super().load_sharded_model(model, checkpoint_index_file, strict, use_safetensors, load_sub_module)
        model.update_master_params()


class LowLevelZeroPlugin(DPPluginBase):
    """
    Plugin for low level zero.

    ```python
    from colossalai.booster import Booster
    from colossalai.booster.plugin import LowLevelZeroPlugin

    model, train_dataset, optimizer, criterion = ...
    plugin = LowLevelZeroPlugin()

    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)
    booster = Booster(plugin=plugin)
    model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)
    ```

    Args:
        stage (int, optional): ZeRO stage. Defaults to 1.
        precision (str, optional): precision. Support 'fp16', 'bf16' and 'fp32'. Defaults to 'fp16'.
        initial_scale (float, optional): Initial scale used by DynamicGradScaler. Defaults to 2**32.
        min_scale (float, optional): Min scale used by DynamicGradScaler. Defaults to 1.
        growth_factor (float, optional): growth_factor used by DynamicGradScaler. Defaults to 2.
        backoff_factor (float, optional): backoff_factor used by DynamicGradScaler. Defaults to 0.5.
        growth_interval (float, optional): growth_interval used by DynamicGradScaler. Defaults to 1000.
        hysteresis (float, optional): hysteresis used by DynamicGradScaler. Defaults to 2.
        max_scale (int, optional): max_scale used by DynamicGradScaler. Defaults to 2**32.
        max_norm (float, optional): max_norm used for `clip_grad_norm`. You should notice that you shall not do
            clip_grad_norm by yourself when using ZeRO DDP. The ZeRO optimizer will take care of clip_grad_norm.
        norm_type (float, optional): norm_type used for `clip_grad_norm`.
        reduce_bucket_size_in_m (int, optional): grad reduce bucket size in M. Defaults to 12.
        communication_dtype (torch.dtype, optional): communication dtype. If not specified, the dtype of param will be used. Defaults to None.
        overlap_communication (bool, optional): whether to overlap communication and computation. Defaults to True.
        cpu_offload (bool, optional): whether to offload grad, master weight and optimizer state to cpu. Defaults to False.
        verbose (bool, optional): verbose mode. Debug info including grad overflow will be printed. Defaults to False.
    """

    def __init__(
        self,
        stage: int = 1,
        precision: str = "fp16",
        initial_scale: float = 2**32,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0.0,
        norm_type: float = 2.0,
        reduce_bucket_size_in_m: int = 12,
        communication_dtype: Optional[torch.dtype] = None,
        overlap_communication: bool = True,
        cpu_offload: bool = False,
        master_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        assert stage in (1, 2), f"LowLevelZeroPlugin only supports stage 1/2 training"
        assert precision in SUPPORTED_PRECISION, f"LowLevelZeroPlugin only supports {SUPPORTED_PRECISION} training"
        assert norm_type == 2.0, f"LowLevelZeroPlugin only supports norm_type=2.0 now"
        self.stage = stage
        self.precision = precision
        self.zero_optim_kwargs = dict(
            initial_scale=initial_scale,
            min_scale=min_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            max_scale=max_scale,
            clip_grad_norm=max_norm,
            reduce_bucket_size=reduce_bucket_size_in_m * 1024 * 1024,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
            partition_grad=(stage == 2),
            cpu_offload=cpu_offload,
            master_weights=master_weights,
        )
        self.verbose = verbose

        # set class name with stage, for better error message
        setattr(self.__class__, "__name__", f"LowLevelZeroPlugin_ZeRO-{stage}")

    def support_no_sync(self) -> bool:
        return self.stage == 1

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return SUPPORTED_PRECISION

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ["cuda", "npu"]

    def configure(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        if not isinstance(model, ModelWrapper):
            model = LowLevelZeroModel(model, self.precision)

        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            optimizer: LowLevelZeroOptimizer = LowLevelZeroOptimizer(
                optimizer, **self.zero_optim_kwargs, verbose=self.verbose
            )
            # inject update_master_params
            model.update_master_params = MethodType(optimizer.update_master_params, model)
            # Setup optimizers that require global states
        if isinstance(optimizer.optim, DistributedOptimizer):
            tp_group = self.__dict__.get("tp_group", None)
            dp_group = self.__dict__.get("dp_group", None)
            master_to_working_map = optimizer.get_master_to_working_map()
            optimizer.optim.setup_distributed(master_to_working_map, tp_group, dp_group, zero_flag=True)

        return model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return LowLevelZeroCheckpointIO()

    def no_sync(self, model: nn.Module, optimizer: OptimizerWrapper) -> Iterator[None]:
        assert isinstance(optimizer, LowLevelZeroOptimizer)
        return optimizer.no_sync()
