import torch

from ...registry import meta_patched_module


@meta_patched_module.register(torch.nn.LayerNorm)
@meta_patched_module.register(torch.nn.GroupNorm)
@meta_patched_module.register(torch.nn.BatchNorm1d)
@meta_patched_module.register(torch.nn.BatchNorm2d)
@meta_patched_module.register(torch.nn.BatchNorm3d)
def torch_nn_normalize(self, input):
    # check shape
    if isinstance(self, torch.nn.BatchNorm1d):
        assert input.dim() in [2, 3]
    elif isinstance(self, torch.nn.BatchNorm2d):
        assert input.dim() == 4
    elif isinstance(self, torch.nn.BatchNorm3d):
        assert input.dim() == 5

    # normalization maintain the same shape as the input
    return input.clone()


try:
    import apex

    meta_patched_module.register(apex.normalization.FusedLayerNorm)(torch_nn_normalize)
    meta_patched_module.register(apex.normalization.FusedRMSNorm)(torch_nn_normalize)
    meta_patched_module.register(apex.normalization.MixedFusedLayerNorm)(torch_nn_normalize)
    meta_patched_module.register(apex.normalization.MixedFusedRMSNorm)(torch_nn_normalize)
except (ImportError, AttributeError):
    pass
