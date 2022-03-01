
import contextlib
import copy
import functools
import os
import traceback
from collections import OrderedDict
from enum import Enum, auto
from typing import (Any, Callable, Dict, Generator, List, NamedTuple, Optional,
                    Set, Union)

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.utils import get_current_device
from torch.distributed import ProcessGroup
from colossalai.engine.ophooks import register_ophooks_recursively, BaseOpHook, ShardParamHook
from colossalai.zero.shard_param import ShardParam

class ShardedModelV2(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 process_group: Optional[ProcessGroup] = None,
                 reduce_scatter_process_group: Optional[ProcessGroup] = None
    ):
        r"""
        A demo to reconfigure zero1 shared_model.
        Currently do not consider the Optimizer States.
        """
        super().__init__()
        self.logger = get_dist_logger()

        self.process_group = process_group or gpc.get_group(ParallelMode.DATA)
        self.reduce_scatter_process_group = reduce_scatter_process_group or self.process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)

        # The module has to be placed on GPU
        self.module = module.cuda()

        # Shard the parameters at first
        for _, param in self.module.named_parameters():
            param.ca_attr = ShardParam(param)
            param.ca_attr.shard()

        # Register hooks
        register_ophooks_recursively(self.module, [ShardParamHook()])

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs


    def backward(self, loss):
        if self.loss_scaler:
            self.loss_scaler.backward(loss)
        else:
            loss.backward()
    
    