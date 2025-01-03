import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from packaging import version
from torch.distributed import ProcessGroup

if version.parse(torch.__version__) >= version.parse("1.12.0"):
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

from colossalai.checkpoint_io import CheckpointIndexFile, CheckpointIO, GeneralCheckpointIO, utils
from colossalai.checkpoint_io.utils import async_save_state_dict_shards, create_pinned_state_dict
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.logging import get_dist_logger
from colossalai.utils.safetensors import load_flat

from .dp_plugin_base import DPPluginBase

__all__ = ["TorchFSDPPlugin"]


class TorchFSDPCheckpointIO(GeneralCheckpointIO):
    def __init__(self) -> None:
        super().__init__()
        self.coordinator = DistCoordinator()
        self.logger = get_dist_logger()

    def load_unsharded_model(
        self, model: ModelWrapper, checkpoint: str, strict: bool, low_cpu_mem_mode: bool = True, num_threads: int = 1
    ):
        assert isinstance(model, TorchFSDPModel), "Please boost the model before loading!"
        model = model.unwrap()
        checkpoint = utils.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint)

    def load_unsharded_optimizer(
        self, optimizer: OptimizerWrapper, checkpoint: Path, low_cpu_mem_mode: bool = True, num_threads: int = 1
    ):
        assert isinstance(optimizer, FSDPOptimizerWrapper), "Please boost the optimizer before loading!"
        if checkpoint.endswith(".safetensors"):
            checkpoint = load_flat(checkpoint, seperator=".")
        else:
            checkpoint = utils.load_state_dict(checkpoint)

        fsdp_model = optimizer.unwrap_model()
        full_optimizer_state = FSDP.full_optim_state_dict(fsdp_model, optim=optimizer, rank0_only=False)
        start_index = 0
        id2name = {}

        def get_index_mapping(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            start_num = len(id2name)
            id2name.update({i: p for i, p in enumerate(group["params"], start_index) if i not in id2name})
            end_num = len(id2name)
            start_index += end_num - start_num

        for g in full_optimizer_state["param_groups"]:
            get_index_mapping(g)

        new_state = {}
        for key, value in checkpoint["state"].items():
            new_state[id2name[int(key)]] = value
        checkpoint["state"] = new_state
        for g in checkpoint["param_groups"]:
            new_group = []
            for param_id in g["params"]:
                new_group.append(id2name[param_id])
            g["params"] = new_group

        sharded_osd = FSDP.scatter_full_optim_state_dict(checkpoint, fsdp_model)
        optimizer.load_state_dict(sharded_osd)

    def save_unsharded_model(
        self, model: ModelWrapper, checkpoint: str, gather_dtensor: bool, use_safetensors: bool, use_async: bool = False
    ):
        """
        Save model to checkpoint but only on master process.
        """
        assert isinstance(model, TorchFSDPModel), "Please boost the model before saving!"
        model = model.unwrap()
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            full_model_state = model.state_dict()
        if self.coordinator.is_master():
            if use_async:
                from colossalai.utils.safetensors import save

                if id(model) not in self.pinned_state_dicts:
                    self.pinned_state_dicts[id(model)] = create_pinned_state_dict(full_model_state)
                for k, v in full_model_state.items():
                    self.pinned_state_dicts[id(model)][k].copy_(v)
                    full_model_state[k] = self.pinned_state_dicts[id(model)][k]
                writer = save(checkpoint, full_model_state)
                self.async_writers.append(writer)
            else:
                utils.save_state_dict(
                    full_model_state, checkpoint_file_path=checkpoint, use_safetensors=use_safetensors
                )

    def save_unsharded_optimizer(
        self, optimizer: OptimizerWrapper, checkpoint: str, gather_dtensor: bool, use_async: bool = False
    ):
        """
        Save optimizer to checkpoint but only on master process.
        """
        assert isinstance(optimizer, FSDPOptimizerWrapper), "Please boost the optimizer before saving!"
        fsdp_model = optimizer.unwrap_model()

        full_optimizer_state = FSDP.full_optim_state_dict(fsdp_model, optim=optimizer, rank0_only=True)

        if self.coordinator.is_master():

            # Save order indices instead of Tensors
            name2id: Dict[str, int] = {}
            start_index = 0

            def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal start_index
                packed = {k: v for k, v in group.items() if k != "params"}
                name2id.update({p: i for i, p in enumerate(group["params"], start_index) if p not in name2id})
                packed["params"] = [name2id[p] for p in group["params"]]
                start_index += len(packed["params"])
                return packed

            param_groups = [pack_group(g) for g in full_optimizer_state["param_groups"]]
            full_optimizer_state["param_groups"] = param_groups
            new_state = {}
            for key, value in full_optimizer_state["state"].items():
                new_state[name2id[key]] = value
            full_optimizer_state["state"] = new_state

            if use_async:
                from colossalai.utils.safetensors import _flatten_optim_state_dict, save

                flatten_state_dict, metadata = _flatten_optim_state_dict(full_optimizer_state, seperator=".")
                if id(optimizer) not in self.pinned_state_dicts:
                    self.pinned_state_dicts[id(optimizer)] = create_pinned_state_dict(flatten_state_dict)
                for k, v in flatten_state_dict.items():
                    self.pinned_state_dicts[id(optimizer)][k].copy_(v)
                    flatten_state_dict[k] = self.pinned_state_dicts[id(optimizer)][k]
                writer = save(checkpoint, state_dict=flatten_state_dict, metadata=metadata)
                self.async_writers.append(writer)
            else:
                utils.save_state_dict(full_optimizer_state, checkpoint_file_path=checkpoint, use_safetensors=False)

    def save_sharded_model(
        self,
        model: ModelWrapper,
        checkpoint_path: str,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
        use_async: bool = False,
    ):
        """
        Save model to checkpoint but only on master process.
        """
        assert isinstance(model, TorchFSDPModel), "Please boost the model before saving!"
        if os.path.isfile(checkpoint_path):
            self.logger.error(f"Provided path ({checkpoint_path}) should be a directory, not a file")
            return

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        with FSDP.state_dict_type(
            model.unwrap(), StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            state_dict = model.unwrap().state_dict()

        if use_async and self.coordinator.is_master():
            if id(model) not in self.pinned_state_dicts:
                self.pinned_state_dicts[id(model)] = {}
            pinned_state_dicts = self.pinned_state_dicts[id(model)]
        else:
            pinned_state_dicts = None
        state_dict_shard = utils.shard_model_checkpoint(
            state_dict, max_shard_size=size_per_shard, pinned_state_dicts=pinned_state_dicts
        )

        weights_name, save_index_file = utils.get_model_base_filenames(prefix, use_safetensors)
        index_file = CheckpointIndexFile(checkpoint_path)

        # In general cases, is_master is set to True to get the right behavior.
        if use_async:
            total_size, writers = async_save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint_path,
                index_file=index_file,
                base_filename=weights_name,
                is_master=self.coordinator.is_master(),
            )
            self.async_writers.extend(writers)
        else:
            total_size = utils.save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint_path,
                index_file=index_file,
                base_filename=weights_name,
                is_master=self.coordinator.is_master(),
                use_safetensors=use_safetensors,
            )

        # only save the index file on the master rank
        if self.coordinator.is_master():
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            utils.save_config_file(model.unwrap(), checkpoint_path)
            self.logger.info(
                f"The model is split into checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def load_sharded_model(
        self,
        model: nn.Module,
        checkpoint_index_file: Path,
        strict: bool = False,
        use_safetensors: bool = False,
        load_sub_module: bool = True,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
        """
        Load model to checkpoint but only on master process.
        """
        assert isinstance(model, TorchFSDPModel), "Please boost the model before loading!"
        use_safetensors = False
        if "safetensors" in checkpoint_index_file.name:
            use_safetensors = True

        if use_safetensors and not utils.is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors` library: `pip install safetensors`.")

        # read checkpoint index file
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()

        fsdp_state_dict = {}
        for state_dict in utils.load_state_dict_shards(checkpoint_files, False, use_safetensors):
            fsdp_state_dict.update(state_dict)

        with FSDP.state_dict_type(model.unwrap(), StateDictType.FULL_STATE_DICT):
            model.unwrap().load_state_dict(fsdp_state_dict, strict=False)

    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: str,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
        use_async: bool = False,
    ):
        """
        Save optimizer to checkpoint but only on master process.
        """
        assert isinstance(optimizer, FSDPOptimizerWrapper), "Please boost the optimizer before saving!"

        if os.path.isfile(checkpoint):
            self.logger.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        with FSDP.state_dict_type(
            optimizer.unwrap_model().unwrap(),
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            fsdp_optim_state = FSDP.full_optim_state_dict(
                optimizer.unwrap_model().unwrap(), optim=optimizer, rank0_only=True
            )

        if self.coordinator.is_master():

            # Save order indices instead of Tensors
            name2id: Dict[str, int] = {}
            start_index = 0

            def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
                nonlocal start_index
                packed = {k: v for k, v in group.items() if k != "params"}
                name2id.update({p: i for i, p in enumerate(group["params"], start_index) if p not in name2id})
                packed["params"] = [name2id[p] for p in group["params"]]
                start_index += len(packed["params"])
                return packed

            param_groups = [pack_group(g) for g in fsdp_optim_state["param_groups"]]
            fsdp_optim_state["param_groups"] = param_groups
            new_state = {}
            for key, value in fsdp_optim_state["state"].items():
                new_state[name2id[key]] = value
            fsdp_optim_state["state"] = new_state

            # Preparing file paths and index file.
            states_name, save_index_file, param_group_file = utils.get_optimizer_base_filenames(
                prefix, use_safetensors=use_async
            )
            index_file = CheckpointIndexFile(checkpoint)

            index_file.append_meta_data("param_groups", param_group_file)
            group_file_path = os.path.join(checkpoint, param_group_file)
            utils.save_param_groups(fsdp_optim_state, group_file_path)

            if use_async:
                if id(optimizer) not in self.pinned_state_dicts:
                    self.pinned_state_dicts[id(optimizer)] = {}
                pinned_state_dicts = self.pinned_state_dicts[id(optimizer)]
            else:
                pinned_state_dicts = None
            sharded_state = utils.shard_optimizer_checkpoint(
                fsdp_optim_state, max_shard_size=size_per_shard, pinned_state_dicts=pinned_state_dicts
            )
            # Save shards of optimizer states.
            # In general cases, is_master is set to True to get the right behavior.
            if use_async:
                total_size, writers = async_save_state_dict_shards(
                    sharded_state_dict=sharded_state,
                    checkpoint=checkpoint,
                    index_file=index_file,
                    base_filename=states_name,
                    is_master=self.coordinator.is_master(),
                    state_preprocess=True,
                )
                self.async_writers.extend(writers)
            else:
                total_size = utils.save_state_dict_shards(
                    sharded_state_dict=sharded_state,
                    checkpoint=checkpoint,
                    index_file=index_file,
                    base_filename=states_name,
                    is_master=self.coordinator.is_master(),
                    use_safetensors=False,
                )

            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)
            self.logger.info(
                f"The optimizer is going to be split to checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

    def load_sharded_optimizer(
        self,
        optimizer: Optimizer,
        index_file_path: str,
        size_per_shard: int,
        low_cpu_mem_mode: bool = True,
        num_threads: int = 1,
    ):
        """
        Load optimizer to checkpoint but only on master process.
        """
        assert isinstance(optimizer, FSDPOptimizerWrapper), "Please boost the optimizer before saving!"

        ckpt_index_file = CheckpointIndexFile.from_file(index_file_path)

        # Load param_groups
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(
                f"Invalid index file path {index_file_path} for an optimizer. "
                "Looking param group file under current directory."
            )

        saved_param_groups = torch.load(param_group_path)

        # Load param
        fsdp_optim_state = {}
        checkpoint_files, _ = ckpt_index_file.get_checkpoint_filenames()
        for state_dict_shard in utils.load_state_dict_shards(checkpoint_files, True, False):
            fsdp_optim_state.update(state_dict_shard)

        fsdp_optim_dict = dict(state=fsdp_optim_state, param_groups=saved_param_groups)

        fsdp_model = optimizer.unwrap_model()
        full_optimizer_state = FSDP.full_optim_state_dict(fsdp_model.unwrap(), optim=optimizer, rank0_only=False)
        start_index = 0
        id2name = {}

        def get_index_mapping(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            start_num = len(id2name)
            id2name.update({i: p for i, p in enumerate(group["params"], start_index) if i not in id2name})
            end_num = len(id2name)
            start_index += end_num - start_num

        for g in full_optimizer_state["param_groups"]:
            get_index_mapping(g)

        new_state = {}
        for key, value in fsdp_optim_dict["state"].items():
            new_state[id2name[int(key)]] = value
        fsdp_optim_dict["state"] = new_state
        for g in fsdp_optim_dict["param_groups"]:
            new_group = []
            for param_id in g["params"]:
                new_group.append(id2name[param_id])
            g["params"] = new_group

        with FSDP.state_dict_type(optimizer.unwrap_model().unwrap(), StateDictType.FULL_STATE_DICT):
            fsdp_state = FSDP.optim_state_dict_to_load(
                model=optimizer.unwrap_model().unwrap(), optim=optimizer, optim_state_dict=fsdp_optim_dict
            )
            optimizer.load_state_dict(fsdp_state)

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

    ```python
    from colossalai.booster import Booster
    from colossalai.booster.plugin import TorchFSDPPlugin

    model, train_dataset, optimizer, criterion = ...
    plugin = TorchFSDPPlugin()

    train_dataloader = plugin.prepare_train_dataloader(train_dataset, batch_size=8)
    booster = Booster(plugin=plugin)
    model, optimizer, train_dataloader, criterion = booster.boost(model, optimizer, train_dataloader, criterion)
    ```

    Args:
        See https://pytorch.org/docs/stable/fsdp.html for details.
    """

    if version.parse(torch.__version__) >= version.parse("1.12.0"):

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
            fp8_communication: bool = False,
        ):
            super().__init__()
            self.fsdp_kwargs = dict(
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                ignored_modules=ignored_modules,
                param_init_fn=param_init_fn,
                sync_module_states=sync_module_states,
            )
            self.fp8_communication = fp8_communication
            self.logger = get_dist_logger()

    else:
        raise RuntimeError("FSDP is not supported while torch version under 1.12.0.")

    def support_no_sync(self) -> bool:
        return False

    def support_lora(self) -> bool:
        return False

    def no_sync(self, model: nn.Module, optimizer: OptimizerWrapper) -> Iterator[None]:
        raise NotImplementedError("Torch fsdp no_sync func not supported yet.")

    def control_precision(self) -> bool:
        return True

    def supported_precisions(self) -> List[str]:
        return ["fp16", "bf16"]

    def control_device(self) -> bool:
        return True

    def supported_devices(self) -> List[str]:
        return ["cuda"]

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

        if self.fp8_communication:
            from colossalai.quantization.utils import patch_fsdp_params_comm_hook

            patch_fsdp_params_comm_hook()

            from colossalai.quantization.fp8 import fp8_compress_fsdp_params_comm_hook

            fsdp_model.module.register_params_comm_hook(None, fp8_compress_fsdp_params_comm_hook)

            from colossalai.quantization.fp8 import fp8_compress_fsdp_grad_comm_hook

            fsdp_model.module.register_comm_hook(None, fp8_compress_fsdp_grad_comm_hook)

        if optimizer is not None:
            if len(optimizer.param_groups) > 1:
                self.logger.warning(
                    "TorchFSDPPlugin does not support optimizer that use multi param groups. The results may not be as expected if used."
                )
            optimizer.__init__(fsdp_model.parameters(), **optimizer.defaults)

            if not isinstance(optimizer, FSDPOptimizerWrapper):
                optimizer = FSDPOptimizerWrapper(optimizer, fsdp_model)

        return fsdp_model, optimizer, criterion, dataloader, lr_scheduler

    def control_checkpoint_io(self) -> bool:
        return True

    def get_checkpoint_io(self) -> CheckpointIO:
        return TorchFSDPCheckpointIO()

    def enable_lora(
        self, model: nn.Module, pretrained_dir: Optional[str] = None, lora_config: Optional[Dict] = None
    ) -> nn.Module:
        raise NotImplementedError
