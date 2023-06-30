import torch.nn as nn

from colossalai.cluster import DistCoordinator

from ..policies.basepolicy import Policy
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

    colossalai.launch_from_torch(config={})

    org_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    shard_config = ShardConfig(
        tensor_parallel_size=2,
        tensor_parallel_mode='1d',
    )
    shard_former = ShardFormer(shard_config=shard_config)
    model = shard_former.shard_model(org_model)
    ```
    """

    def __init__(self, shard_config: ShardConfig):
        """
        Do two things:
        1. Create a colossalai.cluster.process_group_manager to manage process groups for dp, tp and pp
        2. serve as a store for
        """
        self.coordinator = DistCoordinator()
        self.shard_config = shard_config

    def shard_model(self, model: nn.Module, policy: Policy = None):
        r"""
        The function is used to shard the PyTorch model.

        Args:
            model (`torch.nn.Model`): the origin huggingface model
            shard_config (`ShardConfig`): the config for distribute information
            policy (`Policy`): the custom policy for sharding
        """
        sharder = ModelSharder(model=model, shard_config=self.shard_config, policy=policy)
        sharder.shard()
        return model
