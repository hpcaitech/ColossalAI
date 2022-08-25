from typing import Tuple, Union
import torch
from ..registry import meta_profiler_module


@meta_profiler_module.register(torch.nn.InstanceNorm1d)
@meta_profiler_module.register(torch.nn.InstanceNorm2d)
@meta_profiler_module.register(torch.nn.InstanceNorm3d)
@meta_profiler_module.register(torch.nn.LayerNorm)
@meta_profiler_module.register(torch.nn.GroupNorm)
@meta_profiler_module.register(torch.nn.BatchNorm1d)
@meta_profiler_module.register(torch.nn.BatchNorm2d)
@meta_profiler_module.register(torch.nn.BatchNorm3d)
def torch_nn_normalize(self: Union[torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                   torch.nn.BatchNorm3d], input: torch.Tensor) -> Tuple[int, int]:
    # adopted from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/profiling/flops_profiler/profiler.py#L615
    has_affine = self.weight is not None
    if self.training:
        flops = input.numel() * (2 if has_affine else 1)
    else:
        flops = input.numel() * (5 if has_affine else 4)
    macs = 0
    return flops, macs


try:
    import apex
    meta_profiler_module.register(apex.normalization.FusedLayerNorm)(torch_nn_normalize)
    meta_profiler_module.register(apex.normalization.FusedRMSNorm)(torch_nn_normalize)
    meta_profiler_module.register(apex.normalization.MixedFusedLayerNorm)(torch_nn_normalize)
    meta_profiler_module.register(apex.normalization.MixedFusedRMSNorm)(torch_nn_normalize)
except (ImportError, AttributeError):
    pass
