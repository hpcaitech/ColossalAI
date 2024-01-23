import copy
import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import Iterator, Optional, OrderedDict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.checkpoint_io import CheckpointIndexFile
from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import HybridParallelCheckpointIO
from colossalai.checkpoint_io.index_file import CheckpointIndexFile
from colossalai.checkpoint_io.utils import (
    StateDictSharder,
    gather_distributed_param,
    get_model_base_filenames,
    save_config_file,
    save_state_dict_shards,
)
from colossalai.interface import ModelWrapper
from colossalai.moe import MOE_MANAGER

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"


class MixtralMoEHybridParallelCheckpointIO(HybridParallelCheckpointIO):
    def __init__(
        self,
        dp_group: ProcessGroup,
        pp_group: ProcessGroup,
        tp_group: ProcessGroup,
        zero_stage: int,
        verbose: bool = True,
    ) -> None:
        super().__init__(dp_group, pp_group, tp_group, zero_stage, verbose)
        moe_info = MOE_MANAGER.parallel_info_dict[MOE_MANAGER.ep_size]
        self.ep_size = moe_info.ep_size
        self.ep_rank = moe_info.ep_rank
        self.real_dp_rank = moe_info.dp_rank

    @staticmethod
    def _model_sharder(
        model: nn.Module,
        prefix: str = "",
        keep_vars: bool = False,
        size_per_shard: int = 1024,
        param_name_pattern: Optional[str] = None,
    ) -> Iterator[Tuple[OrderedDict, int]]:
        # An internel method that breaks state_dict of model into shards within limited size.

        state_dict_sharder = StateDictSharder(size_per_shard)

        # Save parameters.
        for name, param in model.named_parameters():
            if param is None:
                continue
            if param_name_pattern is not None and param_name_pattern not in name:
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
        if (
            getattr(model.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            extra_state = model.get_extra_state()
            block, block_size = state_dict_sharder.append_param(extra_state_key, extra_state)
            if block is not None:
                yield block, block_size

        # Return the last block in sharder.
        yield state_dict_sharder.current_block, state_dict_sharder.current_block_size

    def save_sharded_model(
        self,
        model: ModelWrapper,
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

        assert isinstance(model, ModelWrapper), "Please boost the model before saving!"
        model = model.unwrap()

        if os.path.isfile(checkpoint):
            logging.error(f"Provided path ({checkpoint}) should be a directory, not a file")
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        if self.real_dp_rank != 0:
            return

        # ep_rank 0 saves all the parameters and buffers.
        # other ep_ranks save only experts
        ep_param_pattern = "experts." if self.ep_rank != 0 else None

        # Then collect the sharded parameters & buffers along tp_group.
        # Only devices with tp_rank == 0 are responsible for model saving.
        state_dict_shard = MixtralMoEHybridParallelCheckpointIO._model_sharder(
            model, size_per_shard=size_per_shard, param_name_pattern=ep_param_pattern
        )
        weights_name, save_index_file = get_model_base_filenames(prefix, use_safetensors)
        index_file = CheckpointIndexFile(checkpoint)
        control_saving = self.tp_rank == 0

        if self.pp_size == 1:
            # When pipeline is not used, save the model shards as in general checkpointIO
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
                if self.verbose and self.coordinator.is_master():
                    logging.info(
                        f"The model is split into checkpoint shards. "
                        f"You can find where each parameters has been saved in the "
                        f"index located at {save_index_file}."
                    )

        else:
            # When pipeline is used, each stage produces its own shard files and index files.
            # Index files belonging to each stage are saved under a temporary folder ./tmp_index_files/
            # After all the state_dicts have been saved, the master rank integrates all the index files into one final index file and deletes the tmp folder.

            final_index_file_path = copy.deepcopy(save_index_file)
            tmp_index_file_folder = os.path.join(checkpoint, "tmp_index_files")
            Path(tmp_index_file_folder).mkdir(parents=True, exist_ok=True)

            # Manage filenames of sharded weights and index file for each pipeline stage.
            weights_name = weights_name.replace(".bin", f"-stage-{self.pp_rank+1:05d}-{self.ep_rank+1:05d}-shard.bin")
            weights_name = weights_name.replace(
                ".safetensors", f"-stage-{self.pp_rank+1:05d}-{self.ep_rank+1:05d}-shard.safetensors"
            )
            save_index_file = save_index_file.replace(".json", f"-stage-{self.pp_rank+1:05d}-{self.ep_rank+1:05d}.json")
            save_index_file = os.path.join("tmp_index_files", save_index_file)

            total_size = save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint,
                index_file=index_file,
                base_filename=weights_name,
                is_master=control_saving,
                use_safetensors=use_safetensors,
                use_pp_format=True,
            )
            if control_saving:
                index_file.append_meta_data("total_size", total_size)
                index_file.write_index_file(save_index_file)
            else:
                return

            dist.barrier(self.pp_group)

            # The global master rank integrates the index files and clean the folder.
            if self.coordinator.is_master():
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
                if self.verbose and self.coordinator.is_master():
                    logging.info(
                        f"The model is split into checkpoint shards. "
                        f"You can find where each parameters has been saved in the "
                        f"index located at {final_index_file_path}."
                    )

    def save_unsharded_model(self, model: ModelWrapper, checkpoint: str, gather_dtensor: bool, use_safetensors: bool):
        raise NotImplementedError
