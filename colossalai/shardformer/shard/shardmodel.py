import os
from contextlib import suppress
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers

from colossalai.tensor.d_tensor.layout import Layout

from ..policies.basepolicy import Policy
from .shardconfig import ShardConfig
from .sharder import ModelSharder


class ShardModel(object):
    r"""
    The class for sharding the huggingface model, ''self.model'' is the sharded model
    Just creat a new ShardModel object to shard huggingface model

    Args:
        model (:class:`torch.nn.Model`): the origin huggingface model
        dist_config (:class:`ShardConfig`): the config for distribute information
        custom_policy (:class:`Policy`): the custom policy for sharding
    """

    def __init__(
        self,
        model: nn.Module,
        shard_config: ShardConfig = None,    # TODO
        custom_policy: Policy = None,
    ) -> None:
        self.model = model
        self.shard_config = shard_config
        self.policy = custom_policy
        # self.layout=,  # TODO

        sharder = ModelSharder(
            model=self.model,
            policy=self.policy,
            shard_config=self.shard_config,
        )
        sharder.shard()

    def set_environ(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "GNU"
        os.environ["MASTER_ADDR"] = str(self.dist_config.master_addr)
        os.environ["MASTER_PORT"] = str(self.dist_config.master_port)
        os.environ["WORLD_SIZE"] = str(self.dist_config.num_gpus)
        os.environ["RANK"] = str(self.dist_config.rank)
        os.environ["LOCAL_RANK"] = str(self.dist_config.rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=self.dist_config.backend)

        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))

    def back_to_org() -> None:
        pass
