from typing import Any, Callable, Dict, List, Optional
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from loralib.layers import LoRALayer
from coati.models.lora import LoraLinear


@dataclass
class LoRAConfig:
    r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0
    fan_in_fan_out: bool = False


class LoRAConstructor:
    '''
    Tools for reconstructing a model from a remote LoRA model.
    (Transferring only LoRA data costs much less!)
    Usage:
        Step 1 (Sender):
            filter_state_dict_lora()
            
        Step 2 (Sender, Optional):
            extract_lora_config()
            
        Step 3 (Sender):
            send state_dict_lora and lora_config_dict
            
        Step 4 (Receiver):
            reconstruct_increase()
            
        Step 5 (Receiver):
            load_state_dict_increase()
            
    '''

    def __init__(self):
        self.lora_config_dict = None

    def register_lora_config(self, lora_config_dict: Dict[str, Any]):
        self.lora_config_dict = lora_config_dict

    def reconstruct_increase(self, state_dict_lora: Dict[str, Any], lora_config_dict: Dict[str, Any]):
        '''
            xxx.lora_A, xxx.lora_B -->> xxx.weight
            Warning: the xxx.weight here is the increment actually.
        '''
        if lora_config_dict is not None:
            self.register_lora_config(lora_config_dict)

        state_dict_increase = OrderedDict()
        config_iter = iter(self.lora_config_dict.items())
        lora_A, lora_B, layer_prefix = None, None, None
        for k, v in state_dict_lora.items():
            if k.rpartition('.')[-1] == 'lora_A':
                lora_A = v
                layer_prefix = k.rpartition('.')[0]
            elif k.rpartition('.')[-1] == 'lora_B':
                assert layer_prefix == k.rpartition('.')[0], "unmatched (lora_A, lora_B) pair"
                layer_prefix_2, config = next(config_iter)
                assert layer_prefix_2 == layer_prefix, "unmatched (state_dict, config_dict) pair"
                lora_B = v
                weight_data_increase = self._compute(lora_A, lora_B, config)
                state_dict_increase[layer_prefix + '.weight'] = weight_data_increase
                lora_A, lora_B, layer_prefix = None, None, None
            else:
                raise ValueError('unexpected key')
        return state_dict_increase

    def _compute(self, lora_A, lora_B, config=LoRAConfig()):
        def T(w):
            return w.T if config.fan_in_fan_out else w
        if config.r > 0:
            scaling = config.lora_alpha / config.r
            weight_data_increase = T(lora_B @ lora_A) * scaling
            return weight_data_increase
        return 0

    def load_state_dict_increase(self, model: nn.Module, state_dict_increase: Dict[str, Any]):
        '''
        The final reconstruction step
        '''
        # naive approach
        model.load_state_dict({k: v + model.state_dict()[k] for k, v in state_dict_increase.items()}, strict=False)

    @staticmethod
    def filter_state_dict_lora(state_dict: Dict[str, Any], keep_non_lora=False):
        '''
        if keep_non_lora, also return non_lora state_dict
        '''
        state_dict_lora = OrderedDict()
        state_dict_non_lora = OrderedDict()
        for k, v in state_dict.items():
            if 'lora_A' in k or 'lora_B' in k:
                state_dict_lora[k] = v
            elif keep_non_lora:
                state_dict_non_lora[k] = v
        if keep_non_lora:
            return state_dict_lora, state_dict_non_lora
        else:
            return state_dict_lora, None

    @staticmethod
    def extract_lora_config(model: nn.Module) -> Dict[str, LoRAConfig]:
        '''
        extract LoraLinear model.
        return OrderedDict(): name -> LoRAConfig
        '''
        lora_config_dict = OrderedDict()

        for name, child in model.named_modules():
            if isinstance(child, LoraLinear):
                lora_config_dict[name] = LoRAConfig(r=child.r,
                                                    lora_alpha=child.lora_alpha,
                                                    lora_dropout=child.lora_dropout,
                                                    fan_in_fan_out=child.fan_in_fan_out)

        return lora_config_dict
