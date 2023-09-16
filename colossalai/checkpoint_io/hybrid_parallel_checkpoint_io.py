import copy
import gc
import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import Dict, Iterator, Optional, OrderedDict, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from colossalai.cluster import DistCoordinator
from colossalai.interface import OptimizerWrapper

from .general_checkpoint_io import GeneralCheckpointIO
from .index_file import CheckpointIndexFile
from .utils import (
    StateDictSharder,
    gather_distributed_param,
    get_model_base_filenames,
    get_optimizer_base_filenames,
    is_safetensors_available,
    load_shard_state_dict,
    load_state_dict_into_model,
    load_states_into_optimizer,
    save_config_file,
    save_param_groups,
    save_state_dict_shards,
    search_tp_partition_dim,
    sharded_optimizer_loading_epilogue,
)

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX, _IncompatibleKeys
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = '_extra_state'


class HybridParallelCheckpointIO(GeneralCheckpointIO):
    """
    CheckpointIO for Hybrid Parallel Training.

    Args:
        dp_group (ProcessGroup): Process group along data parallel dimension.
        pp_group (ProcessGroup): Process group along pipeline parallel dimension.
        tp_group (ProcessGroup): Process group along tensor parallel dimension.
        zero_stage (int): The zero stage of plugin. Should be in [0, 1, 2].
        verbose (bool, optional): Whether to print logging massage when saving/loading has been succesfully executed. Defaults to True.
    """

    def __init__(self,
                 dp_group: ProcessGroup,
                 pp_group: ProcessGroup,
                 tp_group: ProcessGroup,
                 zero_stage: int,
                 verbose: bool = True) -> None:
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
        self.use_zero = (zero_stage > 0)
        self.verbose = verbose
        self.working_to_master_map = None
        self.master_to_working_map = None
        self.coordinator = DistCoordinator()

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
            block, block_size = state_dict_sharder.append_param(prefix + name, param_)
            if block is not None:
                yield block, block_size

        # Save buffers.
        for name, buf in model.named_buffers():
            if buf is not None and name not in model._non_persistent_buffers_set:
                buffer = buf if keep_vars else buf.detach()
                block, block_size = state_dict_sharder.append_param(prefix + name, buffer)
                if block is not None:
                    yield block, block_size

        # Save extra states.
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(model.__class__, "get_extra_state",
                   torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
            extra_state = model.get_extra_state()
            block, block_size = state_dict_sharder.append_param(extra_state_key, extra_state)
            if block is not None:
                yield block, block_size

        # Return the last block in sharder.
        yield state_dict_sharder.current_block, state_dict_sharder.current_block_size

    @staticmethod
    def _optimizer_sharder(optimizer: OptimizerWrapper,
                           use_zero: bool,
                           dp_group: ProcessGroup,
                           tp_group: ProcessGroup,
                           master_to_working_map: Optional[Dict[int, torch.Tensor]] = None,
                           size_per_shard: int = 1024):

        # An internel method that breaks state_dict of optimizer into shards within limited size.

        state_dict_sharder = StateDictSharder(size_per_shard)
        param_info = optimizer.param_info

        for param, state in optimizer.optim.state.items():

            if param is None:
                continue

            if master_to_working_map is not None:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param

            param_id = param_info['param2id'][id(working_param)]
            original_shape = param_info['param2shape'][id(working_param)]
            state_ = HybridParallelCheckpointIO.gather_from_sharded_optimizer_state(state,
                                                                                    working_param,
                                                                                    original_shape=original_shape,
                                                                                    dp_group=dp_group,
                                                                                    tp_group=tp_group,
                                                                                    use_zero=use_zero,
                                                                                    inplace=False)

            block, block_size = state_dict_sharder.append_optim_state(param_id, state_)
            if block is not None:
                yield block, block_size

        # Return the last block in sharder.
        yield state_dict_sharder.current_block, state_dict_sharder.current_block_size

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
        # Only devices with tp_rank == 0 are responsible for model saving.
        state_dict_shard = HybridParallelCheckpointIO._model_sharder(model, size_per_shard=size_per_shard)
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
                save_config_file(model, checkpoint)
                if self.verbose:
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
            weights_name = weights_name.replace(".bin", f"-stage-{self.pp_rank+1:05d}-shard.bin")
            weights_name = weights_name.replace(".safetensors", f"-stage-{self.pp_rank+1:05d}-shard.safetensors")
            save_index_file = save_index_file.replace(".json", f"-stage-{self.pp_rank+1:05d}.json")
            save_index_file = os.path.join("tmp_index_files", save_index_file)

            total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard,
                                                checkpoint=checkpoint,
                                                index_file=index_file,
                                                base_filename=weights_name,
                                                is_master=control_saving,
                                                use_safetensors=use_safetensors,
                                                use_pp_format=True)
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
                save_config_file(model, checkpoint)
                rmtree(tmp_index_file_folder)
                if self.verbose:
                    logging.info(f"The model is split into checkpoint shards. "
                                 f"You can find where each parameters has been saved in the "
                                 f"index located at {final_index_file_path}.")

    def load_sharded_model(self, model: nn.Module, checkpoint_index_file: Path, strict: bool = False):
        """
        Load sharded model with the given path to index file of checkpoint folder.

        Args:
            model (nn.Module): The model to be loaded.
            checkpoint_index_file (str): Path to the index file of checkpointing folder.
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
            loaded_file.add(filename)

        # Load parameters.
        for name, _ in model.named_parameters():
            _load(name)

        # Load buffers.
        non_persistent_buffers = set()
        for n, m in model.named_modules():
            non_persistent_buffers |= set('.'.join((n, b)) for b in m._non_persistent_buffers_set)
        for name, buf in model.named_buffers():
            if buf is not None and name not in non_persistent_buffers:
                _load(name)

        # Load extra states.
        extra_state_key = _EXTRA_STATE_KEY_SUFFIX
        if getattr(model.__class__, "get_extra_state",
                   torch.nn.Module.get_extra_state) is not torch.nn.Module.get_extra_state:
            _load(extra_state_key)

        # Update master params if mixed-precision training is enabled.
        with torch.no_grad():
            if self.working_to_master_map is not None:
                for param in model.parameters():
                    if (param is None) or (id(param) not in self.working_to_master_map):
                        continue
                    master_param = self.working_to_master_map[id(param)]
                    if self.use_zero:
                        # master_param is sharded under Zero setting
                        padding_size = (self.dp_size - param.numel() % self.dp_size) % self.dp_size
                        if padding_size > 0:
                            padded_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
                        else:
                            padded_param = param.data.view(-1)
                        sharded_param = padded_param.split(padded_param.numel() // self.dp_size)[self.dp_rank]
                        master_param.data.copy_(sharded_param.data)
                    else:
                        master_param.data.copy_(param.data)

        if self.verbose:
            logging.info(f"The model has been successfully loaded from sharded checkpoint: {ckpt_root_path}.")

    def save_sharded_optimizer(self,
                               optimizer: OptimizerWrapper,
                               checkpoint: str,
                               gather_dtensor: bool = True,
                               prefix: Optional[str] = None,
                               size_per_shard: int = 1024):
        """
        Save sharded optimizer checkpoint under the given checkpointing path.
        The following files will be created under the path:
        - An index file (pytorch_optim.bin.index.json) containing a map between optimizer states and file names
        - A group file (pytorch_optim_group.bin) recording information of param_groups
        - Multiple files that store state tensors of optimizers.
          If pipeline parallelism is used, the filenames are in the form of "pytorch_optim.<prefix>-stage-000XX-shard-000XX.bin".
          If pipeline parallelism is not used, "pytorch_optim.<prefix>-000XX.bin"

        Args:
            optimizer (OptimizerWrapper): Optimizer to save sharded state_dict
            checkpoint (str): Path to save optimizer state_dict
            gather_dtensor (bool): Whether to gather_dtensor, not used
            prefix (str): Perfix of file to save
            size_per_shard (int): Max file size of each file shard that store state tensors
        """
        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        # Devices along the same dp_group share the same copies of states when zero is not used.
        # In this case only let the device with dp_rank == 0 save the model.
        if not self.use_zero and self.dp_rank != 0:
            return

        # Then collect the sharded states along dp_group(if using zero)/tp_group.
        # Only devices with (dp_rank == 0 and tp_rank == 0) are responsible for states saving.
        state_dict_shard = HybridParallelCheckpointIO._optimizer_sharder(
            optimizer,
            use_zero=self.use_zero,
            dp_group=self.dp_group,
            tp_group=self.tp_group,
            master_to_working_map=self.master_to_working_map,
            size_per_shard=size_per_shard)
        states_name, save_index_file, param_group_file = get_optimizer_base_filenames(prefix)
        index_file = CheckpointIndexFile(checkpoint)
        control_saving = (self.dp_rank == 0 and self.tp_rank == 0)

        if self.pp_size == 1:
            # When pipeline is not used, save the optimizer shards as in general checkpointIO
            total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard,
                                                checkpoint=checkpoint,
                                                index_file=index_file,
                                                base_filename=states_name,
                                                is_master=control_saving)

            if control_saving:
                # Store param groups.
                index_file.append_meta_data("param_groups", param_group_file)
                group_file_path = os.path.join(checkpoint, param_group_file)
                save_param_groups(optimizer.param_info, group_file_path)
                # Store index file.
                index_file.append_meta_data("total_size", total_size)
                index_file.write_index_file(save_index_file)
                if self.verbose:
                    logging.info(f"The optimizer is going to be split to checkpoint shards. "
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
            states_name = states_name.replace(".bin", f"-stage-{self.pp_rank+1:05d}-shard.bin")
            save_index_file = save_index_file.replace(".json", f"-stage-{self.pp_rank+1:05d}.json")
            save_index_file = os.path.join("tmp_index_files", save_index_file)

            total_size = save_state_dict_shards(sharded_state_dict=state_dict_shard,
                                                checkpoint=checkpoint,
                                                index_file=index_file,
                                                base_filename=states_name,
                                                is_master=control_saving,
                                                use_pp_format=True)

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
                    for param_id, state_filename in stage_index_file.weight_map.items():
                        final_index_file.append_weight_map(param_id, state_filename)

                # Store param groups.
                final_index_file.append_meta_data("param_groups", param_group_file)
                group_file_path = os.path.join(checkpoint, param_group_file)
                save_param_groups(optimizer.param_info, group_file_path)

                final_index_file.write_index_file(final_index_file_path)
                rmtree(tmp_index_file_folder)

                if self.verbose:
                    logging.info(f"The model is split into checkpoint shards. "
                                 f"You can find where each parameters has been saved in the "
                                 f"index located at {final_index_file_path}.")

    def load_sharded_optimizer(self, optimizer: OptimizerWrapper, checkpoint_index_file: str, prefix: str = ""):
        """
        Load sharded optimizer with the given path to index file of checkpoint folder.

        Args:
            optimizer (OptimizerWrapper): The optimizer to be loaded.
            checkpoint_index_file (str): Path to the index file of checkpointing folder.
            prefix (str): Not used.
        """

        def _get_param_id_from_optimizer_param(param: torch.Tensor,
                                               master_to_working_map: Optional[Dict[int, torch.Tensor]] = None):
            if master_to_working_map is not None:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param
            return optimizer.param_info['param2id'][id(working_param)]

        # id_map is a mapping from param ids kept by current pipeline, to their corresponding parameter objects.
        # When Zero is used, the mapped parameter objects should be fp32 master parameters.
        # IDs should be obtained through saved param2id mapping earlier saved in optimizer.param_info.
        id_map = {}
        for pg in optimizer.optim.param_groups:
            for param in pg['params']:
                param_id = _get_param_id_from_optimizer_param(param, self.master_to_working_map)
                id_map[param_id] = param

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
        ckpt_root_path = ckpt_index_file.root_path
        weight_map = ckpt_index_file.weight_map
        weight_map = {int(k): v for k, v in weight_map.items()}    # convert saved id from str to int

        # Load param_groups
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(f'Invalid index file path {checkpoint_index_file} for an optimizer. \
                               Lacking param group file under current directory.')
        saved_groups = torch.load(param_group_path)

        updated_groups = []
        for old_pg, saved_pg in zip(optimizer.optim.param_groups, saved_groups):
            # obtain updated param group
            new_pg = copy.deepcopy(saved_pg)
            new_pg['params'] = old_pg['params']    # The parameters in the same group shouln't change.
            updated_groups.append(new_pg)
        optimizer.optim.__dict__.update({'param_groups': updated_groups})

        # Load saved states to optimizer.
        # Keep a record of loaded files so that file will not be repeatedly loaded.
        loaded_file = set()
        for pg in optimizer.optim.param_groups:
            for param in pg['params']:
                if param is None:
                    continue
                param_id = _get_param_id_from_optimizer_param(param, self.master_to_working_map)
                if param_id not in weight_map:
                    continue
                filename = weight_map[param_id]

                # If this param's states has been loaded before, directly return.
                if filename in loaded_file:
                    continue

                file_path = os.path.join(ckpt_root_path, filename)
                state_dict = load_shard_state_dict(Path(file_path), use_safetensors=False)
                load_states_into_optimizer(optimizer.optim, state_dict, id_map, strict=True)
                loaded_file.add(filename)

        # Then shard the loaded optimizer states if using tp/zero.
        for param, state in optimizer.optim.state.items():
            device = param.device
            if self.master_to_working_map is not None:
                working_param = self.master_to_working_map[id(param)]
            else:
                working_param = param
            original_shape = optimizer.param_info['param2shape'][id(working_param)]
            sharded_state = self.shard_from_complete_optimizer_state(state,
                                                                     current_shape=working_param.shape,
                                                                     original_shape=original_shape,
                                                                     device=device,
                                                                     inplace=True)
            optimizer.optim.state[param] = sharded_state

        sharded_optimizer_loading_epilogue(optimizer.optim)
        if self.verbose:
            logging.info(f"The optimizer has been successfully loaded from sharded checkpoint: {ckpt_root_path}.")

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

    def link_master_and_working_param(self, working_to_master_map: Dict[Union[int, torch.Tensor], torch.Tensor],
                                      master_to_working_map: Dict[Union[int, torch.Tensor], torch.Tensor]):
        """
        Create mappings between working params (for forward/backward) and master params (for optimizer update) with passed in mappings.
        This mapping can only be created when mixied precision is used.
        The created mappings should be mappings from integer parameter addresses to parameter objects.

        Args:
            working_to_master_map (Dict[Union[int, torch.Tensor], torch.Tensor]): A mapping from working parameters objects/addresses to master parameter objects.
            master_to_working_map (Dict[Union[int, torch.Tensor], torch.Tensor]): A mapping from master parameters objects/addresses to working parameter objects.
        """
        self.working_to_master_map = dict()
        for k, v in working_to_master_map.items():
            if isinstance(k, torch.Tensor):
                self.working_to_master_map[id(k)] = v
            elif isinstance(k, int):
                self.working_to_master_map[k] = v
            else:
                raise ValueError(
                    f"The passed in mapping should have keys of type 'int' or 'torch.Tensor', but got {type(k)}!")

        self.master_to_working_map = dict()
        for k, v in master_to_working_map.items():
            if isinstance(k, torch.Tensor):
                self.master_to_working_map[id(k)] = v
            elif isinstance(k, int):
                self.master_to_working_map[k] = v
            else:
                raise ValueError(
                    f"The passed in mapping should have keys of type 'int' or 'torch.Tensor', but got {type(k)}!")

    @staticmethod
    def gather_from_sharded_optimizer_state(state: OrderedDict, param: torch.Tensor, original_shape: torch.Size,
                                            dp_group: ProcessGroup, tp_group: ProcessGroup, use_zero: bool,
                                            inplace: bool) -> OrderedDict:
        """
        With given parameter and its optimizer states, gather the complete optimizer state for saving.

        Args:
            state (OrderedDict): Optimizer states of given parameter, might be distributed among tp/dp group if using TP/Zero.
            param (torch.Tensor): The given parameter. It should be working_param when using Zero.
            original_shape (torch.Size): The size of parameter before sharding.
            dp_group (ProcessGroup): The process group of data parallel.
            tp_group (ProcessGroup): The process group of tensor parallel.
            use_zero (bool): Whether Zero is used.
            inplace (bool): If set to True, will update the values of argument 'state' in place. Else will make a copy of state.

        Returns:
            OrderedDict: The complete optimizer state of given parameter.
        """
        dp_size = dist.get_world_size(dp_group)
        tp_size = dist.get_world_size(tp_group)
        current_shape = param.shape
        state_ = state if inplace else copy.deepcopy(state)

        for k, v in state_.items():
            if isinstance(v, torch.Tensor) and k != 'step':

                # First gather Zero shards.
                if use_zero:
                    v = v.cuda()
                    gather_tensor = [torch.zeros_like(v) for _ in range(dp_size)]
                    dist.all_gather(gather_tensor, v, group=dp_group)
                    v = torch.stack(gather_tensor).view(-1)[:param.numel()].reshape_as(param)

                # Then gather TP shards.
                partition_dim = search_tp_partition_dim(current_shape, original_shape, tp_size)
                if partition_dim is not None:
                    gather_tensor = [torch.zeros_like(v) for _ in range(tp_size)]
                    dist.all_gather(gather_tensor, v, group=tp_group)
                    v = torch.cat(gather_tensor, dim=partition_dim)

                state_[k] = v.detach().clone().cpu()

        return state_

    def shard_from_complete_optimizer_state(self, state: OrderedDict, current_shape: torch.Size,
                                            original_shape: torch.Size, device: torch.device,
                                            inplace: bool) -> OrderedDict:
        """
        With complete optimizer states of a specific parameter loaded from checkpoint,
        slice out the sharded optimizer states kept by current device.

        Args:
            state (OrderedDict): Complete optimizer states of a given parameter, loaded from checkpoint.
            current_shape (torch.Size): The size of parameter after sharding.
            original_shape (torch.Size): The size of parameter before sharding.
            device (torch.device): The destination device of loaded optimizer states.
            inplace (bool): If set to True, will update the values of argument 'state' in place. Else will make a copy of state.

        Returns:
            OrderedDict: The sharded optimizer state of the given parameter.
        """
        state_ = state if inplace else copy.deepcopy(state)

        for k, v in state_.items():
            if isinstance(v, torch.Tensor) and k != 'step':

                # Shard state along tensor parallel group.
                partition_dim = search_tp_partition_dim(current_shape, original_shape, self.tp_size)
                if partition_dim is not None:
                    slice_size = current_shape[partition_dim]
                    v = v.split(slice_size, dim=partition_dim)[self.tp_rank]

                # Shard state along data parallel group when using Zero.
                if self.use_zero:
                    padding_size = (self.dp_size - v.numel() % self.dp_size) % self.dp_size
                    with torch.no_grad():
                        v = v.flatten()
                        if padding_size > 0:
                            v = torch.nn.functional.pad(v, [0, padding_size])
                        slice_size = v.numel() // self.dp_size
                        v = v.split(slice_size, dim=0)[self.dp_rank]

                state_[k] = v.detach().clone().to(device)

        return state_
