from typing import Dict, List, Tuple

import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from colossalai.cluster import DistCoordinator

from ..policies.base_policy import Policy
from .shard_config import ShardConfig
from .sharder import ModelSharder


class ShardFormer:
    """
    Parallelize model based on the given config and policy

    Example:

    ```python
    from colossalai.shardformer import ShardFormer, ShardConfig
    from transformers import BertForMaskedLM
    import colossalai
    import torch

    colossalai.launch_from_torch()

    org_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    shard_config = ShardConfig()
    shard_former = ShardFormer(shard_config=shard_config)
    model, shared_params = shard_former.optimize(org_model)
    ```
    """

    def __init__(self, shard_config: ShardConfig):
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.coordinator = DistCoordinator()
        else:
            self.coordinator = None
        self.shard_config = shard_config

    def optimize(self, model: nn.Module, policy: Policy = None) -> Tuple[nn.Module, List[Dict[int, Tensor]]]:
        r"""
        This method will optimize the model based on the given policy.

        Args:
            model (`torch.nn.Model`): the origin huggingface model
            shard_config (`ShardConfig`): the config for distribute information
            policy (`Policy`): the custom policy for sharding

        Returns: the sharded model and the shared parameters
        """
        sharder = ModelSharder(model=model, shard_config=self.shard_config, policy=policy)
        shared_params = sharder.shard()
        return model, shared_params
