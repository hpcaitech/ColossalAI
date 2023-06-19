#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Union

import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.global_variables import tensor_parallel_env as env
from colossalai.nn import init as init

Fast_LN = None
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNorm
    Fast_LN = FastLayerNorm
except ImportError:
    pass


class ParallelModule(nn.Module, ABC):

    @abstractmethod
    def from_native_module(module: nn.Module,
                           process_group: Union[ProcessGroup, List[ProcessGroup]] = None) -> "ParallelModule":
        """
        Convert a native PyTorch module to a parallelized module.

        Args:
            module (nn.Module): the module to be converted.
            process_group (ProcessGroup or list[ProcessGroup]): the process group(s) to be used for communication.
                If this is a list, the process group at the ith index of the list will correspond to the process group
                in the ith axis of the device mesh. Defaults to None, which means the global process group.
        """
        pass
