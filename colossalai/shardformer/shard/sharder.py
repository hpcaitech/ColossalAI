from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from ..policies.autopolicy import get_autopolicy
from ..policies.basepolicy import Col_Layer, Dropout_Layer, Policy, Row_Layer, Embedding_Layer
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
        self.reshape_embedding()
        self.inject_model(self.model)
        self.replace_layer(self.model)
        self.bind_layer(self.model)

    def reshape_embedding(self,) -> None:
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
        vocab_size = self.model_config.vocab_size
        world_size = self.shard_config.world_size
        if vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size
            self.model.resize_token_embeddings(new_vocab_size)
            self.model_config = self.model.config

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
        if inject_policy is None:
            return

        if inject_policy is None:
            return
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
            self.traverse_replace_layer(model, origin_layer_cls, attr_dict, param_funcs)

    def traverse_replace_layer(
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
        if layer.__class__ == origin_cls:
            for k, v in attr_dict.items():
                setattr_(layer, k, v, ignore=True)
            self.shard_one_layer(layer, param_funcs)
        for name, child in layer.named_children():
            self.traverse_replace_layer(child, origin_cls, attr_dict, param_funcs)
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
        for func in param_funcs:
            policy_layers = func()
            for policy_layer in policy_layers:
                suffix = policy_layer.suffix
                replace_layer_cls = policy_layer.replace_layer
                ignore = policy_layer.ignore
                reversed = policy_layer.reversed
                n_cast = policy_layer.n_cast

                assert replace_layer_cls is not None, 'replace_layer should not be None'

                # create new object to replace the origin layer
                # Linear
                suffix_layer = getattr_(org_layer, suffix, ignore=True)
                assert suffix_layer is not None or ignore, f"Layer {org_layer.__class__.__qualname__} has no attribute {suffix}"
                if suffix_layer is None and ignore:
                    continue
                if isinstance(policy_layer, (Col_Layer, Row_Layer, Embedding_Layer)):
                    weight = None
                    bias = None
                    weight_attr = suffix + '.' + policy_layer.weight if policy_layer.weight is not None else None
                    bias_attr = suffix + '.' + policy_layer.bias if hasattr(policy_layer, 'bias') and policy_layer.bias is not None else None

                    if weight_attr is not None:
                        if hasattr_(org_layer, weight_attr):
                            weight = getattr_(org_layer, weight_attr)
                        else:
                            raise ValueError(f"Layer {org_layer.__class__.__qualname__} has no attribute {weight_attr}")

                    if bias_attr is not None:
                        if hasattr_(org_layer, bias_attr):
                            bias = getattr_(org_layer, bias_attr)
                        else:
                            raise ValueError(f"Layer {org_layer.__class__.__qualname__} has no attribute {bias_attr}")

                    # set the sliced weight and bias to the new nn_col layer
                    assert weight is not None or bias is not None

                    # slice weight and bias
                    weight, bias = self.slicer.slice_weight_bias(weight, bias, policy_layer.__class__, n_cast, reversed)

                    if replace_layer_cls.__name__ == "Linear1D_Row":
                        replace_layer = replace_layer_cls(weight.shape[1],
                                                          weight.shape[0],
                                                          bias=False if bias is None else True)
                    elif replace_layer_cls.__name__ == "Linear1D_Col":
                        gather_output = policy_layer.gather_output and self.shard_config.gather_output
                        replace_layer = replace_layer_cls(weight.shape[0],
                                                          weight.shape[1],
                                                          bias=False if bias is None else True,
                                                          gather_output=gather_output)
                    elif replace_layer_cls.__name__ == "Embedding1D":
                        gather_output = policy_layer.gather_output
                        replace_layer = replace_layer_cls(weight.shape[0],
                                                          weight.shape[1],
                                                          gather_output=gather_output)
                    elif replace_layer_cls.__name__ == "VocabParallelEmbedding1D":
                        replace_layer = replace_layer_cls(weight.shape[0], weight.shape[1],
                                                          getattr_(org_layer, f"{suffix}.padding_idx", ignore=True))
                        # setattr_(org_layer, suffix, replace_layer, ignore=ignore)
                        # self.set_param(replace_layer, weight, bias)
                    else:
                        raise NotImplementedError(
                            f"Replacing to {replace_layer_cls.__name__} is not implemented so far")
                    setattr_(org_layer, suffix, replace_layer, ignore=ignore)
                    self.set_param(replace_layer, weight, bias)
                # dropout
                elif isinstance(policy_layer, Dropout_Layer):
                    p_attr = suffix + '.' + policy_layer.p
                    p = getattr_(org_layer, p_attr, ignore=True)
                    replace_layer = replace_layer_cls(p)
                    setattr_(org_layer, suffix, replace_layer, ignore=ignore)
                else:
                    raise NotImplementedError(
                        f"Replacing {getattr_(org_layer, suffix).__class__} is not implemented so far")

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
        if binding_map is None:
            return
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
