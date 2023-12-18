import logging
import os
from pathlib import Path

import torch.distributed as dist
import torch.nn as nn

from colossalai.checkpoint_io import CheckpointIndexFile
from colossalai.checkpoint_io.utils import is_safetensors_available, load_shard_state_dict, load_state_dict_into_model
from colossalai.moe import MoECheckpintIO
from colossalai.tensor.moe_tensor.api import get_ep_rank, get_ep_size, is_moe_tensor


class MixtralMoECheckpointIO(MoECheckpintIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pre_load_model(self, model: nn.Module, state_dict: dict) -> dict:
        """
        Preprocess state_dict before loading and slice the state_dict of MOE tensors.
        """
        model_param_dict = dict(model.named_parameters())
        for name, param in list(state_dict.items()):
            if ".experts." in name:
                if ".experts.gate.weight" in name:
                    new_name = name.replace(".experts.gate.weight", ".experts.gate_weight")
                    state_dict[new_name] = state_dict.pop(name)
                else:
                    str_idx = name.index(".experts.")
                    int(name.split(".")[-3])
                    if ".w1." in name:
                        model_param_name = name.replace(name[str_idx:], ".experts.wi_gate")
                    elif ".w2." in name:
                        model_param_name = name.replace(name[str_idx:], ".experts.wi_up")
                    elif ".w3." in name:
                        model_param_name = name.replace(name[str_idx:], ".experts.wo")
                    model_param = model_param_dict[model_param_name]
                    assert is_moe_tensor(model_param)

                    ep_rank = get_ep_rank(model_param)
                    ep_size = get_ep_size(model_param)
                    expert_num = 8 // ep_size
                    range(ep_rank * expert_num, (ep_rank + 1) * expert_num)

                    state_dict[name] = param

        for name, param in list(state_dict.items()):
            new_name = "module." + name
            state_dict[new_name] = state_dict.pop(name)
            assert new_name in model_param_dict, f"{new_name} not in model"
        dist.barrier()
        return state_dict

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
            name = name.replace("module.", "")
            name = name.replace(".gate_weight", ".gate.weight")
            if ".experts.wi_gate" in name:
                for i in range(8):
                    new_name = name.replace(".experts.wi_gate", f".experts.{i}.w1.weight")
                    _load(new_name)
            elif ".experts.wi_up" in name:
                for i in range(8):
                    new_name = name.replace(".experts.wi_up", f".experts.{i}.w3.weight")
                    _load(new_name)
            elif ".experts.wo" in name:
                for i in range(8):
                    new_name = name.replace(".experts.wo", f".experts.{i}.w2.weight")
                    _load(new_name)
            else:
                _load(name)

        if self.verbose:
            logging.info(f"The model has been successfully loaded from sharded checkpoint: {ckpt_root_path}.")
