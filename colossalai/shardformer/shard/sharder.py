import torch.nn as nn
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from .shardmodel import ShardConfig
from policies.basepolicy import Policy, Layer
from policies.autopolicy import get_autopolicy
from .slicer import Slicer

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
            dist_config: ShardConfig, # TODO
        ) -> None:
        self.model = model
        self.policy = get_autopolicy(self.model) if policy is None else policy
        self.slicer = Slicer()

    def shard(self) -> None:
        self.replace_model()
        self.replace_layer(self.model)
        
    def replace_model(self) -> None:
        """
        Replace the model to policy defined model
        Mainly modify the forward and backward to fit distributed model
        e.g.:
            BertForMaskedLM -> BertForMaskedLM_
        """
        pass

    def replace_layer(self, layer: nn.Module) -> None:
        """
        Replace the layer according to the policy

        Args:
            layer: The layer to shard
        """
        pass

    def shard_layer(self, policy: Policy) -> nn.Module:
        """
        Shard the layer's weight and bias according to the policy
        
        Args:
            policy
        
        Returns:
            The sharded layer: nn.Module
        """
        pass