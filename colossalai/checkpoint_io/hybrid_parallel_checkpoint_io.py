import copy
import gc
import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Iterator, List, Optional, OrderedDict, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from colossalai.cluster import ProcessGroupMesh
from colossalai.tensor.d_tensor import (
    is_customized_distributed_tensor,
    is_distributed_tensor,
    to_global,
    to_global_for_customized_distributed_tensor,
)

from .general_checkpoint_io import GeneralCheckpointIO
from .index_file import CheckpointIndexFile
from .utils import (
    StateDictSharder,
    calculate_tensor_size,
    gather_distributed_param,
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

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, _IncompatibleKeys
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'


class HypridParallelCheckpointIO(GeneralCheckpointIO):
    """
    CheckpointIO for Hybrid Parallel Training.

    Args:
        dp_group (ProcessGroup): Process group along data parallel dimension.
        pp_group (ProcessGroup): Process group along pipeline parallel dimension.
        tp_group (ProcessGroup): Process group along tensor parallel dimension.
    """

    def __init__(self, dp_group: ProcessGroup, pp_group: ProcessGroup, tp_group: ProcessGroup) -> None:
        super().__init__()
        self.dp_group = dp_group
        self.pp_group = pp_group
        self.tp_group = tp_group
        self.dp_rank = dist.get_rank(self.dp_group)
        self.tp_rank = dist.get_rank(self.tp_group)
        self.pp_rank = dist.get_rank(self.pp_group)
        self.dp_size = dist.get_world_size(dp_group)
        self.pp_size = dist.get_world_size(pp_group)
        self.tp_size = dist.get_world_size(tp_group)

    @staticmethod
    def _model_sharder(model: nn.Module,
                       prefix: str = '',
                       keep_vars: bool = False,
                       size_per_shard: int = 1024) -> Iterator[Tuple[OrderedDict, int]]:
        # An internel method that breaks state_dict of model into shards within limited size.

        state_dict_sharder = StateDictSharder(size_per_shard)

        # Save parameters.
        for name, param in model.named_parameters():
            if param is None:
                continue
            # Gather tensor pieces when using tensor parallel.
            param_ = gather_distributed_param(param, keep_vars=False)
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
        - An index file (pytorch_model.bin.index.json) containing a map between model params/buffers and file names.
        - Multiple files that store state tensors of models.
          If pipeline parallelism is used, the filenames are in the form of "pytorch_model.<prefix>-stage-000XX-shard-000XX.bin".
          If pipeline parallelism is not used, "pytorch_model.<prefix>-000XX.bin"


        Args:
            model (nn.Module): Model on local device to be saved.
            checkpoint (str): Checkpointing path which should be a directory path.
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
        ckpt_root_path = ckpt_index_file.root_path
        weight_map = ckpt_index_file.weight_map
        strict = False

        # Load params & buffers to model.
        # Keep a record of loaded files so that file will not be repeatedly loaded.
        loaded_file = set()

        def _load(name: str):
            if name not in weight_map:
                raise ValueError(f"{name} is not stored in checkpoint, please check your checkpointing configuration!")
            filename = weight_map[name]

            # If this param/buffer has been loaded before, directly return.
            if filename in loaded_file:
                return

            file_path = os.path.join(ckpt_root_path, filename)
            state_dict = load_shard_state_dict(Path(file_path), use_safetensors)
            missing_keys = []

            load_state_dict_into_model(model,
                                       state_dict,
                                       missing_keys=missing_keys,
                                       strict=strict,
                                       load_sub_module=True)
            del state_dict
            loaded_file.add(filename)

        # Load parameters.
        for name, _ in model.named_parameters():
            _load(name)

        # Load buffers.
        for name, buf in model.named_buffers():
            if buf is not None and name not in model._non_persistent_buffers_set:
                _load(name)

        # Load extra states.
        extra_state_key = _EXTRA_STATE_KEY_SUFFIX
        if getattr(model.__class__, "get_extra_state",
                   torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
            _load(extra_state_key)

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
