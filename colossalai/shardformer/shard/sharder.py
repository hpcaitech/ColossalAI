from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn

from ..policies.autopolicy import get_autopolicy
from ..policies.basepolicy import Policy
from ..utils.utils import getattr_, hasattr_, setattr_
from .shard_config import ShardConfig
from .slicer import Slicer

__all__ = ['ModelSharder', 'shard_model']


class ModelSharder(object):
    r"""
    Shard the original huggingface model according to the policy

    Args:
        policy (:class:`Policy`): The policy to shard the model
        model (:class:`torch.Module`): The model to shard
        shard_config: The setting of distributed model
    """

    def __init__(
            self,
            model: nn.Module,
            policy: Policy,
            shard_config: ShardConfig = None,    # TODO
    ) -> None:
        self.model = model
        self.policy = get_autopolicy(self.model) if policy is None else policy
        self.slicer = Slicer(shard_config)
        self.shard_config = shard_config
        self.model_config = self.model.config

    def shard(self) -> None:
        self.inject_model(self.model)
        self.replace_layer(self.model)
        self.bind_layer(self.model)

    def inject_model(
        self,
        model: nn.Module,
    ) -> None:
        r"""
        Replace the model to policy defined model
        Mainly modify the forward and backward to fit distributed model

        e.g.
        ::
            BertForMaskedLM.forward -> BertForMaskedLM_.forward
        """
        inject_policy = self.policy.inject_policy()

        org_model_cls = inject_policy[0]
        shard_model_cls = inject_policy[1]

        if model.__class__ == org_model_cls:
            for key in shard_model_cls.__dict__.keys():
                if hasattr(model.__class__, key):
                    setattr(
                        model.__class__,
                        key,
                        getattr(shard_model_cls, key),
                    )
        else:
            raise NotImplementedError(f"{model.__class__} is not implemented so far")

    def replace_layer(
        self,
        model: nn.Module,
    ) -> None:
        r"""
        Replace the layer according to the policy, and replace the layer one by one

        Args:
            model (:class:`torch.nn.Module`): The layer to shard
        """
        argument_policies = self.policy.argument_policy(self.model_config, self.shard_config.world_size)
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
        r"""
        Reverse the replace layer operation

        Args:
            layer (:class:`torch.nn.Module`): The object of layer to shard
            origin_cls (:class:`transformers.model`): The origin layer class
            attr_dict (Dict): The attribute dict to modify
            policy_cls (:class:`Policy`): The policy class
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

    def shard_one_layer(
        self,
        org_layer: nn.Module,
        param_funcs: List[Callable],
    ) -> None:
        r"""
        Shard one layer according to the policy, the layer should be the same class as the key in policy's argument_policy return dict

        Args:
            org_layer (:class:`torch.nn.Module`): The origin layer object to shard
            param_funcs (:class:`List[typing.Callable]`): The function list to get shard information in policy class

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
                if policy_layer.__class__.__name__ == "Col_Layer":
                    gather_output = policy_layer.gather_output
                    # print(gather_output)

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

                # slice weight and bias
                weight, bias = self.slicer.slice_weight_bias(weight, bias, policy_layer.__class__)
                # print(os.environ['RANK'], policy_layer.__class__, weight.shape, bias.shape if bias is not None else None)

                # create new object to replace the origin layer
                if replace_layer_cls is not None:
                    # print(f"RANK {os.environ['RANK']}: replace {getattr_(org_layer, layer_attr).__class__} to {replace_layer_cls}, shape is {weight.shape}")
                    if isinstance(getattr_(org_layer, layer_attr), nn.Linear):
                        if replace_layer_cls.__name__ == "Linear1D_Row":
                            replace_layer = replace_layer_cls(weight.shape[1],
                                                              weight.shape[0],
                                                              bias=False if bias is None else True)
                        elif replace_layer_cls.__name__ == "Linear1D_Col":
                            replace_layer = replace_layer_cls(weight.shape[0],
                                                              weight.shape[1],
                                                              bias=False if bias is None else True,
                                                              gather_output=gather_output)
                        setattr_(org_layer, layer_attr, replace_layer, ignore=ignore)
                        self.set_param(replace_layer, weight, bias)
                    elif isinstance(getattr_(org_layer, layer_attr), nn.Embedding):
                        replace_layer = replace_layer_cls(weight.shape[0], weight.shape[1],
                                                          getattr_(org_layer, f"{layer_attr}.padding_idx", ignore=True))
                        setattr_(org_layer, layer_attr, replace_layer, ignore=ignore)
                        self.set_param(replace_layer, weight, bias)
                    else:
                        raise NotImplementedError(
                            f"Replacing {getattr_(org_layer, layer_attr).__class__} is not implemented so far")
                # do not replace the layer object, just replace the weight and bias
                else:
                    self.set_param(org_layer, layer_attr, weight, bias)

    def set_param(self,
                  layer: Any,
                  weight: torch.Tensor = None,
                  bias: torch.Tensor = None,
                  layer_attr: str = "") -> None:
        r"""
        Reset the weight and bias of the layer object

        Args:
            layer (:class:`torch.nn.Module`): The layer object
            layer_attr (str): The attribute name of the layer
            weight (:class:`torch.Tensor`): The weight of the layer
            bias (:class:`torch.Tensor`): The bias of the layer
        """
        assert weight is not None or bias is not None
        if weight is not None:
            setattr_(layer, "weight" if layer_attr == "" else layer_attr + ".weight", nn.Parameter(weight.contiguous()))
            self.set_layer_size(layer, layer_attr, weight.shape)
        if bias is not None:
            setattr_(layer, "bias" if layer_attr == "" else layer_attr + ".bias", nn.Parameter(bias.contiguous()))

    def set_layer_size(self, layer: nn.Module, layer_attr: str, size: torch.Size) -> None:
        r"""
        Set the layer attribute

        Args:
            layer (:class:`torch.nn.Module`): The layer object
            layer_attr (str): The attribute name of the layer
            size (:class:`torch.Size`): The size of the tensor
        """
        # Tensor.shape[0] -> out_features, Tensor.shape[1] -> in_features
        attrs = ["out_features", "in_features"]
        for i, attr in enumerate(attrs):
            if hasattr_(layer, f"{layer_attr}.{attr}"):
                setattr_(layer, f"{layer_attr}.{attr}", size[i])

    def bind_layer(self, model: nn.Module) -> None:
        r"""
        Bind the layer according to the binding policy

        Args:
            model (:class:`torch.nn.Module`): The shard model
        """
        binding_map = self.policy.binding_policy()
        for k, v in binding_map.items():
            param = getattr_(model, k)
            param = nn.Parameter(param)
            setattr_(model, k, param)
            setattr_(model, v, param)


def shard_model(model: nn.Module, shard_config: ShardConfig = None, policy: Policy = None):
    r"""
    The function is used to shard the PyTorch model.

    Args:
        model (`torch.nn.Model`): the origin huggingface model
        shard_config (`ShardConfig`): the config for distribute information
        policy (`Policy`): the custom policy for sharding
    """
    sharder = ModelSharder(model=model, shard_config=shard_config, policy=policy)
    sharder.shard()
    return model
