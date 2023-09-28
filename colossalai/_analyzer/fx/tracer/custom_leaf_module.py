import torch

from .tracer import register_leaf_module, register_leaf_module_impl

try:
    import apex

    register_leaf_module(apex.normalization.FusedLayerNorm)
    register_leaf_module(apex.normalization.FusedRMSNorm)
    register_leaf_module(apex.normalization.MixedFusedLayerNorm)
    register_leaf_module(apex.normalization.MixedFusedRMSNorm)

    @register_leaf_module_impl(apex.normalization.FusedLayerNorm)
    @register_leaf_module_impl(apex.normalization.FusedRMSNorm)
    @register_leaf_module_impl(apex.normalization.MixedFusedLayerNorm)
    @register_leaf_module_impl(apex.normalization.MixedFusedRMSNorm)
    def torch_nn_normalize(self, input: torch.Tensor):
        # check shape
        if isinstance(self, torch.nn.BatchNorm1d):
            assert input.dim() in [2, 3]
        elif isinstance(self, torch.nn.BatchNorm2d):
            assert input.dim() == 4
        elif isinstance(self, torch.nn.BatchNorm3d):
            assert input.dim() == 5

        # normalization maintain the same shape as the input
        return input.clone()

except (ImportError, AttributeError):
    pass
