import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
# from colossalai.shardformer.shard.shardmodel import ShardConfig
from dataclasses import dataclass
from ..policies.basepolicy import Policy, Layer
from ..policies.autopolicy import get_autopolicy
from .slicer import Slicer
from ..utils.utils import hasattr_, setattr_
import colossalai.nn as col_nn

@dataclass
class ShardConfig:
    """
    The config for sharding the huggingface model for test
    """
    fp16: bool
    num_gpus: int
    rank: int
    backend="nccl"
    verbose: str = 'simple'
    seed: int = None
    require_grad: bool = False
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

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
            model_config,
            dist_config: ShardConfig = None, # TODO
        ) -> None:
        self.model = model
        self.policy = get_autopolicy(self.model) if policy is None else policy
        self.slicer = Slicer()
        self.dist_config = dist_config
        self.model_config = model_config

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
        print(inject_policy)
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
        Replace the layer according to the policy

        Args:
            layer: The layer to shard
        """
        argument_policies = policy_cls.argument_policy(self.model_config, 2)
        for argument_policy in argument_policies.items():
            origin_layer_cls = argument_policy[0]
            attr_dict = argument_policy[1]
            self.reverse_replace_layer(model, origin_layer_cls, attr_dict, policy_cls)

    def shard_layer(self, policy_obj: Policy) -> nn.Module:
        """
        Shard the layer's weight and bias according to the policy
        
        Args:
            policy
        
        Returns:
            The sharded layer: nn.Module
        """
        return None
        pass

    def reverse_replace_layer(
            self,
            layer: nn.Module,
            origin_cls: nn.Module,
            attr_dict: Dict,
            policy_cls: Policy,
        ) -> None:
        """
        Reverse the replace layer operation
        """
        for name, child in layer.named_children():
            if child.__class__ == origin_cls:
                policy_obj = policy_cls(replace_layer=child)

                for k, v in attr_dict.items():
                    setattr_(policy_obj, f"replace_layer.{k}", v, ingore=True)
                setattr_(layer, name, self.shard_layer(policy_obj))

            self.reverse_replace_layer(child, origin_cls, attr_dict, policy_cls)
        return layer