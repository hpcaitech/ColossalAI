import inspect
import types

import torch
from torch import nn


class DiffusionPipe(nn.Module):
    """
    This Class convert a class of `DiffusionPipeline` into `nn.Module` and reserve most of origin attr,function and property.
    """

    def __init__(self, source_obj) -> None:
        super(DiffusionPipe, self).__init__()

        for k, v in source_obj.__dict__.items():
            if isinstance(v, nn.Module):
                self.add_module(k, v)
            else:
                setattr(self, k, v)

        skip_list = ["_execution_device", "to", "device"]  # this

        for name, member in inspect.getmembers(source_obj.__class__):
            if name in skip_list:
                continue
            if not name.startswith("__") and not name.endswith("__"):
                if isinstance(member, property):
                    setattr(self.__class__, name, member)
                elif inspect.isfunction(member) or inspect.ismethod(member):
                    bound_method = types.MethodType(member, self)
                    setattr(self, name, bound_method)
                elif not callable(member) and not isinstance(member, property):
                    setattr(self, name, member)
            elif name == "__call__":
                bound_method = types.MethodType(member, self)
                setattr(self, "_forward", bound_method)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        # return self.device
        return torch.device("cuda")

    @property
    def device(self):
        next(self.parameters()).device

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)
