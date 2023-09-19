import torch

from colossalai.utils.model.utils import call_to_str


class LayerSpec:
    """ """

    def __init__(self, typename, *module_args, **module_kwargs):
        self.typename = typename
        self.module_args = module_args
        self.module_kwargs = module_kwargs
        self.children = None
        self._param_count = 0

        if not issubclass(typename, torch.nn.Module):
            raise RuntimeError("LayerSpec only supports torch.nn.Module types.")

    def __repr__(self):
        return call_to_str(self.typename.__name__, self.module_args, self.module_kwargs)

    @property
    def param_count(self):
        return self._param_count

    def build(self):
        """Build the stored specification."""

        recovered_args = []
        for obj in self.module_args:
            if isinstance(obj, LayerSpec):
                obj = obj.build()
            recovered_args.append(obj)
        recovered_args = tuple(recovered_args)

        recovered_kwargs = {}
        for k, v in self.module_kwargs.items():
            if isinstance(v, LayerSpec):
                v = v.build()
            recovered_kwargs[k] = v

        return self.typename(*recovered_args, **recovered_kwargs)

    def set_children(self, children):
        self.children = children

    def count_params(self):
        self._param_count = 0
        layer = self.build()
        for param in layer.parameters():
            self._param_count += param.numel()
        return self._param_count

    def reset_param_count(self):
        self._param_count = 0
