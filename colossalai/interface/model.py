import re
from typing import Dict, Set

import torch
import torch.nn as nn
from peft import PeftModel, PeftType


def extract_lora_layers(model: PeftModel, names: Set[str], adapter_name: str = "default"):
    config = model.peft_config[adapter_name]
    if config.peft_type != PeftType.LORA:
        raise ValueError(f"Adapter {adapter_name} is not a LORA adapter.")
    # to_return = lora_state_dict(model, bias=model.peft_config.bias)
    # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
    # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
    bias = config.bias
    if bias == "none":
        to_return = {k for k in names if "lora_" in k}
    elif bias == "all":
        to_return = {k for k in names if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = set()
        for k in names:
            if "lora_" in k:
                to_return.add(k)
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in names:
                    to_return.add(bias_name)
    else:
        raise NotImplementedError
    to_return = {k for k in to_return if (("lora_" in k and adapter_name in k) or ("bias" in k))}
    if config.use_dora:
        # Here we take care of a refactor of DoRA which changed lora_magnitude_vector from a ParameterDict to a
        # ModuleDict with a DoraLayer instance. The old parameter is now the "weight" attribute of that layer. Since
        # we want the state_dict format not to change, we remove the "weight" part.
        new_dora_suffix = f"lora_magnitude_vector.{adapter_name}.weight"

        def renamed_dora_weights(k):
            if k.endswith(new_dora_suffix):
                k = k[:-7]  # remove ".weight"
            return k

        to_return = {renamed_dora_weights(k) for k in to_return}

    to_return = {re.sub(f"lora_\S\.{adapter_name}\.(weight|bias)", "base_layer", k) for k in to_return}
    return to_return


class PeftUnwrapMixin:
    def __init__(self, peft_model: PeftModel):
        self.base_model = peft_model.get_base_model()
        # peft does not affect buffers
        self.lora_layers = extract_lora_layers(peft_model, set(n for n, p in self.base_model.named_parameters()))
        potential_lora_weights = set()
        for n in self.lora_layers:
            potential_lora_weights.add(f"{n}.weight")
            potential_lora_weights.add(f"{n}.bias")
        self.lora_param_to_origin_param = {n: n.replace("base_layer.", "") for n in potential_lora_weights}
        self.origin_param_to_lora_param = {v: k for k, v in self.lora_param_to_origin_param.items()}

    def named_parameters(self):
        for n, p in self.base_model.named_parameters():
            if n in self.lora_param_to_origin_param:
                n = self.lora_param_to_origin_param[n]
            yield n, p

    def named_buffers(self):
        return self.base_model.named_buffers()

    @property
    def _modules(self):
        return self.base_model._modules

    @property
    def _non_persistent_buffers_set(self):
        return self.base_model._non_persistent_buffers_set

    def patch_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in self.origin_param_to_lora_param:
                k = self.origin_param_to_lora_param[k]
            new_state_dict[k] = v
        return new_state_dict

    def state_dict(self):
        state_dict = {}
        for k, v in self.base_model.state_dict().items():
            if k in self.lora_param_to_origin_param:
                k = self.lora_param_to_origin_param[k]
            state_dict[k] = v
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = self.patch_state_dict(state_dict)
        self.base_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def __hash__(self):
        return hash(self.base_model)


class ModelWrapper(nn.Module):
    """
    A wrapper class to define the common interface used by booster.

    Args:
        module (nn.Module): The model to be wrapped.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def unwrap(self, unwrap_peft: bool = True):
        """
        Unwrap the model to return the original model for checkpoint saving/loading.
        """
        if isinstance(self.module, ModelWrapper):
            model = self.module.unwrap()
        else:
            model = self.module
        if unwrap_peft and isinstance(model, PeftModel):
            model = PeftUnwrapMixin(model)
        return model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class AMPModelMixin:
    """This mixin class defines the interface for AMP training."""

    def update_master_params(self):
        """
        Update the master parameters for AMP training.
        """
