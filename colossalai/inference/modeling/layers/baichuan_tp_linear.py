from typing import List, Union

import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup

from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer import Linear1D_Col
from colossalai.shardformer.layer.parallel_module import ParallelModule


class BaichuanLMHeadLinear1D_Col(Linear1D_Col):
    @staticmethod
    def from_native_module(
        module: nn.Module, process_group: Union[ProcessGroup, List[ProcessGroup]], *args, **kwargs
    ) -> ParallelModule:
        LazyInitContext.materialize(module)
        module.in_features = module.weight.size(1)
        module.out_features = module.weight.size(0)
        module.bias = None
        module.weight.data = nn.functional.normalize(
            module.weight
        )  # NOTE(lry89757) This behavior may not apply to lazy init. When we use lazy init, the weight of shardformer is not the real weight.
        # So we should rewrite our own load_from_state_dict of `BaichuanLMHeadLinear1D_Col` to fix this potential issue.

        # get the attributes
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device
        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        if out_features < tp_size:
            return module

        if out_features % tp_size != 0:
            raise ValueError(
                f"The size of out_features:{out_features} is not integer multiples of tensor parallel size: {tp_size}!"
            )

        lmhead_1d = BaichuanLMHeadLinear1D_Col(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            **kwargs,
        )

        return lmhead_1d

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        state_dict[prefix + "weight"] = nn.functional.normalize(state_dict[prefix + "weight"])
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
