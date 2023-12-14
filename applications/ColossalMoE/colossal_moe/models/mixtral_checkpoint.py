import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.checkpoint_io import CheckpointIndexFile
from colossalai.checkpoint_io.utils import is_safetensors_available, load_shard_state_dict, load_state_dict_into_model
from colossalai.moe import MoECheckpintIO
from colossalai.tensor.moe_tensor.api import get_dp_rank, get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor


class MixtralMoECheckpointIO(MoECheckpintIO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def pre_load_model(self, model: nn.Module, state_dict: dict) -> dict:
        """
        Preprocess state_dict before loading and slice the state_dict of MOE tensors.
        """
        model_param_dict = dict(model.named_parameters())
        for name, param in list(state_dict.items()):
            if ".gate.weight" in name:
                new_name = "module." + name.replace(".gate.weight", ".gate_weight")
                state_dict[new_name] = state_dict.pop(name)
            elif ".experts." in name:
                # if is moe tensor
                # in our moe module, expert is cat as one tensor
                # but mixtral's experts is not cat
                # we will insert the loaded expert into the position of cat tensor

                # get model param
                str_idx = name.index(".experts.")
                expert_idx = int(name.split(".")[-3])
                if ".w1." in name:
                    model_param_name = name.replace(name[str_idx:], ".experts.wi_gate")
                elif ".w2." in name:
                    model_param_name = name.replace(name[str_idx:], ".experts.wo")
                elif ".w3." in name:
                    model_param_name = name.replace(name[str_idx:], ".experts.wi_up")
                model_param_name = "module." + model_param_name
                # skip for pipeline
                if model_param_name not in model_param_dict:
                    continue
                model_param = model_param_dict[model_param_name]
                assert is_moe_tensor(model_param)
                # get expert range
                ep_rank = get_ep_rank(model_param)
                ep_size = get_ep_size(model_param)
                expert_num = 8 // ep_size
                expert_range = list(range(ep_rank * expert_num, (ep_rank + 1) * expert_num))
                # insert new param
                if expert_idx in expert_range:
                    new_param = model_param
                    new_param[expert_idx - ep_rank * expert_num] = param.transpose(0, 1)
                    state_dict[model_param_name] = new_param
                state_dict.pop(name)
            else:
                new_name = "module." + name
                state_dict[new_name] = state_dict.pop(name)

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

    @torch.no_grad()
    def pre_save_model(self, model: nn.Module) -> dict:
        torch.cuda.empty_cache()
        state_dict = model.state_dict()
        for name, param in list(model.named_parameters()):
            if ".gate_weight" in name:
                new_name = name.replace(".gate_weight", ".gate.weight")
                state_dict[new_name] = state_dict.pop(name).cpu()
            elif ".experts." in name:
                ep_group = get_ep_group(param)
                ep_rank = get_ep_rank(param)
                ep_size = get_ep_size(param)
                dp_rank = get_dp_rank(param)

                if dp_rank == 0:
                    param = param.data.cuda()
                    all_param = [torch.zeros_like(param) for _ in range(ep_size)]
                    # gather param from every ep rank
                    dist.all_gather(all_param, param, group=ep_group)
                    if ep_rank == 0:
                        all_param = torch.cat(all_param, dim=0)
                        assert all_param.shape[0] == 8
                        for i in range(8):
                            if ".wi_gate" in name:
                                new_name = name.replace(".experts.wi_gate", f".experts.{i}.w1.weight")
                            elif ".wi_up" in name:
                                new_name = name.replace(".experts.wi_up", f".experts.{i}.w3.weight")
                            elif ".wo" in name:
                                new_name = name.replace(".experts.wo", f".experts.{i}.w2.weight")
                            new_name = new_name.replace("module.", "")
                            new_param = all_param[i].transpose(-1, -2)
                            state_dict[new_name] = new_param.cpu()
                        state_dict.pop(name)
            else:
                state_dict[name] = param.cpu()

        for name, param in list(state_dict.items()):
            new_name = name.replace("module.", "")
            state_dict[new_name] = state_dict.pop(name)
        
        torch.cuda.empty_cache()
        if self.pp_size > 1:
            if self.dp_rank == 0:
                # gather state_dict from every pp rank
                # because ckpt is large, we split it into 10 parts
                # and gather them one by one
                new_state_dict = {}
                state_dict_keys = list(state_dict.keys())
                gap_key_num = min(30, len(state_dict_keys))
                gap_keys = (len(state_dict_keys) + gap_key_num - 1) // gap_key_num
                for i in range(gap_key_num):
                    cur_keys = state_dict_keys[i * gap_keys : (i + 1) * gap_keys]
                    cur_state_dict = {}
                    for k in cur_keys:
                        cur_state_dict[k] = state_dict[k]
                    out = [None for _ in range(self.pp_size)]
                    dist.all_gather_object(out, cur_state_dict, group=self.pp_group)
                    if self.pp_rank == 0:
                        for o in out:
                            for k, v in o.items():
                                new_state_dict[k] = v.cpu()
                state_dict = new_state_dict
        dist.barrier()
        return state_dict
