import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
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
        e.g.:
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
            attr_dict = argument_policy[1]
            self.reverse_replace_layer(model, origin_layer_cls, attr_dict, policy_cls)


    def reverse_replace_layer(
            self,
            layer: nn.Module,
            origin_cls: nn.Module,
            attr_dict: Dict,
            policy_cls: Policy,
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
                self.shard_one_layer(child, policy_cls)
                continue

            self.reverse_replace_layer(child, origin_cls, attr_dict, policy_cls)
        return layer


    def shard_layer(self, policy_obj: Policy) -> nn.Module:
        """
        Shard the layer's weight and bias according to the policy
        
        Args:
            policy
        
        Returns:
            The sharded layer: nn.Module
        """
        attn_inw, attn_inb, attn_inw_attr, attn_inb_attr = self.preprocess(
            policy.attn_in(),
            policy,
        )

        attn_outw, attn_outb, attn_outw_attr, attn_outb_attr = self.preprocess(
            policy.attn_out(),
            policy,
        )
        mlp_inw, mlp_inb, mlp_inw_attr, mlp_inb_attr = self.preprocess(
            policy.mlp_in(),
            policy,
        )
        mlp_outw, mlp_outb, mlp_outw_attr, mlp_outb_attr = self.preprocess(
            policy.mlp_out(),
            policy,
        )
        emd_w, emd_b, emd_w_attr, emd_b_attr = self.preprocess(
            policy.embedding(),
            policy,
        )
        unemd_w, unemd_b, unemd_w_attr, unemd_b_attr = self.preprocess(
            policy.unembedding(),
            policy,
        )

        policy = self.set_parameters(
            policy,
            attn_inw,
            attn_inb,
            *self.slicer.column_slice(
                (attn_inw, attn_inb),
                (attn_inw_attr, attn_inb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            attn_outw,
            attn_outb,
            *self.slicer.row_slice(
                (attn_outw, attn_outb),
                (attn_outw_attr, attn_outb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            mlp_inw,
            mlp_inb,
            *self.slicer.column_slice(
                (mlp_inw, mlp_inb),
                (mlp_inw_attr, mlp_inb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            mlp_outw,
            mlp_outb,
            *self.slicer.row_slice(
                (mlp_outw, mlp_outb),
                (mlp_outw_attr, mlp_outb_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            emd_w,
            emd_b,
            *self.slicer.column_slice(
                (emd_w, emd_b),
                (emd_w_attr, emd_b_attr),
            ),
        )

        policy = self.set_parameters(
            policy,
            unemd_w,
            unemd_b,
            *self.slicer.column_slice(
                (unemd_w, unemd_b),
                (unemd_w_attr, unemd_b_attr),
            ),
        )

        return policy_obj.replace_layer

    def shard_one_layer(self, org_layer: nn.Module, policy: Policy):
        """
        Shard one layer
        """
        # print(org_layer)
        attn_in = policy.attn_in()
        for layer in attn_in:
            weight = None
            bias = None
            weight_attr = layer.weight
            bias_attr = layer.bias
            replace_layer_cls = layer.replace_layer
            ignore = layer.ignore

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

            # dont have the attribute in policy 
            if weight is None and bias is None and ignore:
                continue

            # set the sliced weight and bias to the new nn_col layer
            assert weight is not None or bias is not None
            weight, bias = self.slicer.slice_weight_bias(weight, bias, 0)
            replece_layer = replace_layer_cls(weight.shape[0], weight.shape[1], bias=True)
            # print(replece_layer)
            # replece_layer.weight = nn.Parameter(weight)
            # replece_layer.bias = nn.Parameter(bias)
            setattr_(org_layer, weight_attr[:weight_attr.rfind(".")], replece_layer, ignore=ignore)
    
