import torch
import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, Callable
from .shardconfig import ShardConfig
from dataclasses import dataclass
from ..policies.basepolicy import Policy, Layer
from ..policies.autopolicy import get_autopolicy
from .slicer import Slicer
from ..utils.utils import hasattr_, setattr_, getattr_
import colossalai.nn as col_nn


class ModelSharder(object):
    """
    Shard the original huggingface model according to the policy

    Args:
        policy: The policy to shard the model
        model: The model to shard
        dist_setting: The setting of distributed model
    """
    def __init__(
            self,
            model: nn.Module,
            policy: Policy,
            shard_config: ShardConfig = None, # TODO
        ) -> None:
        self.model = model
        self.policy = get_autopolicy(self.model) if policy is None else policy
        self.slicer = Slicer(shard_config)
        self.shard_config = shard_config
        self.model_config = self.model.config


    def shard(self) -> None:
        self.inject_model(self.model, self.policy)
        self.replace_layer(self.model, self.policy)
  
        
    def inject_model(
            self,
            model: nn.Module,
            policy_cls: Policy
        ) -> None:
        """
        Replace the model to policy defined model
        Mainly modify the forward and backward to fit distributed model
        
        e.g.
            BertForMaskedLM.forward -> BertForMaskedLM_.forward
        """
        inject_methods = ["forward"]
        inject_policy = policy_cls.inject_policy()

        org_model_cls = inject_policy[0]
        shard_model_cls = inject_policy[1]

        if model.__class__ == org_model_cls:
            for inject_method in inject_methods:
                if hasattr(model, inject_method):
                    setattr(
                        model,
                        inject_method,
                        getattr(shard_model_cls,inject_method),
                    )
        else:
            raise NotImplementedError(f"{model.__class__} is not implemented so far")


    def replace_layer(
            self,
            model: nn.Module,
            policy_cls: Policy
        ) -> None:
        """
        Replace the layer according to the policy, and replace the layer one by one

        Args:
            layer: The layer to shard
        """
        argument_policies = policy_cls.argument_policy(self.model_config, 2)
        for argument_policy in argument_policies.items():
            origin_layer_cls = argument_policy[0]
            attr_dict = argument_policy[1].attr_dict
            param_funcs = argument_policy[1].param_funcs
            self.reverse_replace_layer(model, origin_layer_cls, attr_dict, param_funcs)


    def reverse_replace_layer(
            self,
            layer: nn.Module,
            origin_cls: nn.Module,
            attr_dict: Dict[str, Any],
            param_funcs: List[Callable],
        ) -> None:
        """
        Reverse the replace layer operation

        Args:
            layer: The object of layer to shard
            origin_cls: The origin layer class
            attr_dict: The attribute dict to modify
            policy_cls: The policy class
        """
        for name, child in layer.named_children():
            if child.__class__ == origin_cls:
                # replac_layer = child
                for k, v in attr_dict.items():
                    setattr_(child, k, v, ignore=True)
                # print(f"Sharding {name} layer", replac_layer.attention.self.__dict__)
                # setattr_(layer, name, self.shard_one_layer(child, policy_cls))
                self.shard_one_layer(child, param_funcs)
                continue

            self.reverse_replace_layer(child, origin_cls, attr_dict, param_funcs)
        return layer


    def shard_one_layer(self, org_layer: nn.Module, param_funcs: List[Callable]) -> None:
        """
        Shard one layer according to the policy, the layer should be the same class as the key in policy's argument_policy return dict

        Args:
            org_layer: The origin layer object to shard
            param_funcs: The function list to get shard information in policy class

        """
        # print(org_layer)
        for func in param_funcs:
            policy_layers = func()
            for policy_layer in policy_layers:
                weight = None
                bias = None
                weight_attr = policy_layer.weight
                bias_attr = policy_layer.bias
                replace_layer_cls = policy_layer.replace_layer
                ignore = policy_layer.ignore

                if weight_attr is not None:
                    if hasattr_(org_layer, weight_attr):
                        weight = getattr_(org_layer, weight_attr)
                    elif not ignore:
                        raise ValueError(f"Layer {org_layer.__class__.__qualname__} has no attribute {weight_attr}")

                if bias_attr is not None:
                    if hasattr_(org_layer, bias_attr):
                        bias = getattr_(org_layer, bias_attr)
                    elif not ignore:
                        raise ValueError(f"Layer {org_layer.__class__.__qualname__} has no attribute {bias_attr}")

                # dont have the attribute in policy, and ignore is true
                if weight is None and bias is None and ignore:
                    continue

                # set the sliced weight and bias to the new nn_col layer
                assert weight is not None or bias is not None
                layer_attr = (lambda x: x[:x.rfind(".")])(weight_attr or bias_attr)

                weight, bias = self.slicer.slice_weight_bias(weight, bias, policy_layer.__class__)
                
                # create new object to replace the origin layer
                # TODO: col_nn
                if replace_layer_cls is not None:
                    replece_layer = replace_layer_cls(weight.shape[0], weight.shape[1], bias=True)
                    # print(replece_layer)
                    # replece_layer.weight = nn.Parameter(weight)
                    # replece_layer.bias = nn.Parameter(bias)
                    setattr_(org_layer, layer_attr, replece_layer, ignore=ignore)
                # do not replace the layer object, just replace the weight and bias
                else:
                    self.set_param(org_layer, layer_attr, weight, bias)


    def set_param(self, layer: Any, layer_attr: str, weight: torch.Tensor, bias: torch.Tensor = None) -> None:
        """
        Reset the weight and bias of the layer object

        Args:
            layer: The layer object
            layer_attr: The attribute name of the layer
            weight: The weight of the layer
            bias: The bias of the layer
        """
        assert weight is not None or bias is not None
        if weight is not None:
            setattr_(layer, layer_attr+".weight", nn.Parameter(weight))
            self.set_layer_size(layer, layer_attr, weight.shape)
        if bias is not None:
            setattr_(layer, layer_attr+".bias", nn.Parameter(bias))


    def set_layer_size(self, layer: nn.Module, layer_attr: str, size: torch.Size) -> None:
        """
        Set the layer attribute

        Args:
            layer: The layer object
            layer_attr: The attribute name of the layer
            size: Torch.size
        """
        attrs = ["out_features", "in_features"]
        for i, attr in enumerate(attrs):
            if hasattr_(layer, f"{layer_attr}.{attr}"):
                setattr_(layer, f"{layer_attr}.{attr}", size[i])    
