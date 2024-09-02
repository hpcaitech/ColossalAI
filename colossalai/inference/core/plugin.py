import logging
import os
from functools import reduce
from pathlib import Path
from typing import Optional

import torch

from colossalai.checkpoint_io.general_checkpoint_io import GeneralCheckpointIO
from colossalai.checkpoint_io.index_file import CheckpointIndexFile
from colossalai.checkpoint_io.utils import is_safetensors_available, load_shard_state_dict, load_state_dict_into_model
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper

try:
    from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX
except ImportError:
    _EXTRA_STATE_KEY_SUFFIX = "_extra_state"


class InferCheckpoint_io(GeneralCheckpointIO):
    """
    This class is for inference model loading, most codes are copied from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io.HybridParallelCheckpointIO.
    Origin HybridParallelCheckpointIO contains some codes about MixPrecision-Training, so we remove them and build a relatively clean class specifically for Inference.
    """

    def __init__(
        self,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.coordinator = DistCoordinator()

    def load_sharded_model(self, model: ModelWrapper, checkpoint_index_file: Path, strict: bool = False):
        """
        Load sharded model with the given path to index file of checkpoint folder.

        Args:
            model (nn.Module): The model to be loaded.
            checkpoint_index_file (str): Path to the index file of checkpointing folder.
            strict (bool, optional): For name matching during loading state_dict. Defaults to False.
                                     This argument should be manually set to False since params on same device might be stored in different files.
        """
        assert isinstance(model, ModelWrapper), "Please boost the model before loading!"
        model = model.unwrap()

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

        missing_keys = []
        missing_file_keys = []

        def _load(name: str):
            if name not in weight_map:
                missing_file_keys.append(name)
                return
            filename = weight_map[name]

            # If this param/buffer has been loaded before, directly return.
            if filename in loaded_file:
                return

            file_path = os.path.join(ckpt_root_path, filename)
            state_dict = load_shard_state_dict(Path(file_path), use_safetensors)

            load_state_dict_into_model(
                model, state_dict, missing_keys=missing_keys, strict=strict, load_sub_module=True
            )
            loaded_file.add(filename)

        # Load parameters.
        for name, _ in model.named_parameters():
            _load(name)

        # Load buffers.
        non_persistent_buffers = set()
        for n, m in model.named_modules():
            non_persistent_buffers |= set(".".join((n, b)) for b in m._non_persistent_buffers_set)
        for name, buf in model.named_buffers():
            if buf is not None and name not in non_persistent_buffers:
                _load(name)

        # Load extra states.
        extra_state_key = _EXTRA_STATE_KEY_SUFFIX
        if (
            getattr(model.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
            is not torch.nn.Module.get_extra_state
        ):
            _load(extra_state_key)

        if self.verbose and self.coordinator.is_master():
            logging.info(f"The model has been successfully loaded from sharded checkpoint: {ckpt_root_path}.")

        if len(missing_keys) == 0:
            raise RuntimeError(
                "No weigth is loaded into the model. Please check the checkpoint files and the model structure."
            )

        remain_keys = reduce(lambda a, b: a & b, map(set, missing_keys))
        remain_keys = remain_keys.union(set(missing_file_keys))
        if len(remain_keys) > 0:
            if strict:
                error_msgs = [
                    "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys))
                ]
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
            else:
                if self.coordinator.is_master():
                    logging.info(f"The following keys are not loaded from checkpoint: {remain_keys}")

    def save_sharded_model(
        self,
        model: ModelWrapper,
        checkpoint: str,
        gather_dtensor: bool = True,
        prefix: Optional[str] = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
    ) -> None:
        return NotImplementedError
