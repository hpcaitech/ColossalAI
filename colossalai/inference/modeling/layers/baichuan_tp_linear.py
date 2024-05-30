from typing import List, Union

import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.shardformer.layer import Linear1D_Col
from colossalai.shardformer.layer.parallel_module import ParallelModule


class BaichuanLMHeadLinear1D_Col(Linear1D_Col):
    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        module.in_features = module.weight.size(1)
        module.out_features = module.weight.size(0)
        module.bias = None
        module.weight.data = nn.functional.normalize(module.weight)

        return Linear1D_Col.from_native_module(
            module,
            process_group,
            *args,
            **kwargs,
        )


class BaichuanWpackLinear1D_Col(Linear1D_Col):
    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        in_features = module.in_features * 3
        out_features = module.out_features // 3
        module.weight.data = module.weight.view(3, out_features, -1).transpose(0, 1).reshape(out_features, in_features)
        module.bias = None

        return Linear1D_Col.from_native_module(
            module,
            process_group,
            *args,
            **kwargs,
        )
