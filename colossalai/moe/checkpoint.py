import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Optional, OrderedDict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.optim import Optimizer

from colossalai.checkpoint_io import CheckpointIndexFile, HybridParallelCheckpointIO
from colossalai.checkpoint_io.utils import (
    StateDictSharder,
    gather_distributed_param,
    get_model_base_filenames,
    is_safetensors_available,
    load_shard_state_dict,
    load_state_dict_into_model,
    save_config_file,
    save_state_dict_shards,
)
from colossalai.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import get_dp_rank, get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor


class MoeCheckpintIO(HybridParallelCheckpointIO):

    def __init__(
        self,
        dp_group: ProcessGroup,
        pp_group: ProcessGroup,
        tp_group: ProcessGroup,
        zero_stage: int,
    ) -> None:
        assert zero_stage in [
            0,
            1,
            2,
        ], f"zero_stage should be 0 or 1 or 2, got {zero_stage}"
        super().__init__(dp_group, pp_group, tp_group, zero_stage)
        self.parallel = MOE_MANAGER.parallel

    def pre_load_model(self, model: nn.Module, state_dict: dict) -> dict:
        """
        Preprocess state_dict before loading and slice the state_dict of MOE tensors.
        """
        for name, param in state_dict.items():
            if ".experts." in name:
                if name in dict(model.named_parameters()):
                    model_param = dict(model.named_parameters())[name]
                    if is_moe_tensor(model_param):
                        ep_rank = get_ep_rank(model_param)
                        ep_size = get_ep_size(model_param)
                        expert_num = param.shape[0] // ep_size
                        assert param.shape[0] % ep_size == 0
                        param = param[ep_rank * expert_num:(ep_rank + 1) * expert_num]
                        state_dict[name] = param
        dist.barrier()
        return state_dict

    def _model_sharder(
        self,
        state_dict: nn.Module,
        prefix: str = "",
        keep_vars: bool = False,
        size_per_shard: int = 1024,
    ) -> Iterator[Tuple[OrderedDict, int]]:
        # An internel method that breaks state_dict of model into shards within limited size.
        state_dict_sharder = StateDictSharder(size_per_shard)

        for name, param in state_dict.items():
            if param is None:
                continue
            # Gather tensor pieces when using tensor parallel.
            param_ = gather_distributed_param(param, keep_vars=False)
            block, block_size = state_dict_sharder.append_param(prefix + name, param_)
            if block is not None:
                yield block, block_size

        # Return the last block in sharder.
        yield state_dict_sharder.current_block, state_dict_sharder.current_block_size

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool) -> None:
        state_dict = torch.load(checkpoint)
        state_dict = self.pre_load_model(model, state_dict)
        model.load_state_dict(state_dict, strict=strict if self.pp_size == 1 else False)

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
            state_dict = self.pre_load_model(model, state_dict)
            missing_keys = []

            load_state_dict_into_model(
                model,
                state_dict,
                missing_keys=missing_keys,
                strict=strict,
                load_sub_module=True,
            )
            loaded_file.add(filename)

        # Load parameters.
        for name, _ in model.named_parameters():
            _load(name)

        if self.verbose:
            logging.info(f"The model has been successfully loaded from sharded checkpoint: {ckpt_root_path}.")

    def pre_save_model(self, model: nn.Module) -> dict:
        state_dict = model.state_dict()
        for name, param in model.named_parameters():
            if ".experts." in name and is_moe_tensor(param):
                ep_group = get_ep_group(param)
                ep_rank = get_ep_rank(param)
                ep_size = get_ep_size(param)
                dp_rank = get_dp_rank(param)
                if dp_rank == 0:
                    param = param.data.cuda()
                    all_param = [deepcopy(param) for _ in range(ep_size)]
                    # gather param from every ep rank
                    dist.all_gather(all_param, param, group=ep_group)
                    if ep_rank == 0:
                        all_param = torch.cat(all_param, dim=0)
                        state_dict[name] = all_param.cpu()
        if self.pp_size > 1:
            if self.dp_rank == 0:
                out = [None for _ in range(self.pp_size)]
                dist.all_gather_object(out, state_dict, group=self.pp_group)
                if self.pp_rank == 0:
                    new_state_dict = {}
                    for o in out:
                        new_state_dict.update(o)
                    state_dict = new_state_dict
        dist.barrier()
        return state_dict

    def save_unsharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        gather_dtensor: bool,
        use_safetensors: bool,
    ):
        state_dict = self.pre_save_model(model)
        if dist.get_rank() == 0:
            torch.save(state_dict, checkpoint)
        dist.barrier()

    def save_sharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
    ) -> None:
        """
        Save sharded model checkpoint under the given checkpointing path.
        The following files will be created under the path:
        - An index file (pytorch_model.bin.index.json) containing a map between model params/buffers and file names.
        - Multiple files that store state tensors of models.
          The filenames are in the form of "pytorch_model.<prefix>-000XX.bin"

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

        # Then collect the sharded parameters & buffers along tp_group.
        # Only devices with tp_rank == 0 are responsible for model saving.
        state_dict = self.pre_save_model(model)

        if dist.get_rank() == 0:
            state_dict_shard = self._model_sharder(state_dict, size_per_shard=size_per_shard)

            # Devices along the same dp_group share the same copies of model.
            # So only let the device with dp_rank == 0 save the model.
            if self.dp_rank != 0:
                return

            weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
            index_file = CheckpointIndexFile(checkpoint)
            control_saving = self.tp_rank == 0

            total_size = save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint,
                index_file=index_file,
                base_filename=weights_name,
                is_master=control_saving,
                use_safetensors=use_safetensors,
            )
            if control_saving:
                index_file.append_meta_data("total_size", total_size)
                index_file.write_index_file(save_index_file)
                save_config_file(model, checkpoint)
                if self.verbose:
                    logging.info(f"The model is split into checkpoint shards. "
                                 f"You can find where each parameters has been saved in the "
                                 f"index located at {save_index_file}.")
        dist.barrier()

    # ========================================================
    # Abstract methods for optimizer loading/saving implementation
    # ========================================================

    def load_sharded_optimizer(self, optimizer: Optimizer, index_file_path: str, prefix: str):
        raise NotImplementedError()

    def load_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path):
        raise NotImplementedError()

    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
    ):
        raise NotImplementedError()

    def save_unsharded_optimizer(self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool):
        raise NotImplementedError()
