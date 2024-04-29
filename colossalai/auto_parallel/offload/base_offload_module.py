from functools import partial
from typing import Optional, Set

import torch
import torch.nn as nn

from colossalai.utils import _cast_float, get_current_device
from colossalai.utils.common import free_storage

from .region_manager import RegionManager
from .util import GlobalRuntimeInfo


class BaseOffloadModule:
    """
    BaseOffloadModule: A model wrapper for parameter offloading.

    Args:
        model (nn.Module): model to apply offloading.
        region_manager (RegionManager): a ``RegionManager`` instance.
        is_sync (bool): synchronous mode or not.
    """

    def __init__(self, model: nn.Module, region_manager: RegionManager, is_sync=True):
        self.model = model
        self.region_manager = region_manager
        self.grad_hook_list = []
        self.overflow_counter = torch.tensor([0], dtype=torch.int, device=get_current_device())

        self.grad_offload_stream = torch.cuda.current_stream() if is_sync else GlobalRuntimeInfo.d2h_stream

        self._cast_buffers()

    def register_grad_hook(self):
        for p in self.model.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(partial(self.grad_handle, p)))

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        self.register_grad_hook()
        for region in self.region_manager.region_list:
            region.cpu_grad = None

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, torch.half), _cast_float(kwargs, torch.half)
        self.model.zero_grad(set_to_none=True)
        self._pre_forward()
        outputs = self.model(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._post_backward()

    def _post_backward(self):
        torch.cuda.synchronize()
        self.remove_grad_hook()

        for p in self.model.parameters():
            p.grad = None

        GlobalRuntimeInfo().fwd_prefetch_event_map.clear()
        GlobalRuntimeInfo().bwd_prefetch_event_map.clear()

    def grad_handle(self, p, grad):
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        with torch._C.DisableTorchFunction():
            region = self.region_manager.get_region(p)
            region.copy_grad_to_region_slice(p, grad)
            if region.can_release:
                self.overflow_counter += region.has_inf_or_nan
                master_stream = torch.cuda.current_stream()
                with torch.cuda.stream(self.grad_offload_stream):
                    GlobalRuntimeInfo().d2h_stream.wait_stream(master_stream)
                    region.move_grad_to_cpu()
        return empty_grad

    def _cast_buffers(self):
        for buffer in self.model.buffers():
            buffer.data = buffer.cuda()

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix, recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        return self.model.named_buffers(prefix, recurse)

    def named_children(self):
        return self.model.named_children()

    def named_modules(
        self, memo: Optional[Set[torch.nn.Module]] = None, prefix: str = "", remove_duplicate: bool = True
    ):
        return self.model.named_modules(memo, prefix, remove_duplicate)
