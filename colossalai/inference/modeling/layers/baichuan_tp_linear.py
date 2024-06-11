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
        module.weight.data = nn.functional.normalize(
            module.weight
        )  # TODO(lry89757) This behavior may not apply to lazy init. When we use lazy init, the weight of shardformer is not the real weight.
        # So we should rewrite our own load_from_state_dict of `BaichuanLMHeadLinear1D_Col` to fix this potential issue.

        return Linear1D_Col.from_native_module(
            module,
            process_group,
            *args,
            **kwargs,
        )
