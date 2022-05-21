import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from functools import partial

__all__ = ['ColoDDP']


def free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


class ColoDDP(torch.nn.Module):

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.dp_world_size = gpc.get_world_size(ParallelMode.DATA)
        for p in module.parameters():
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle, p))

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)

    def forward(self, *args, **kwargs):
        self.module.zero_grad(set_to_none=True)
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        loss.backward()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        for p in self.module.parameters():
            p.grad = p._saved_grad

    def grad_handle(self, p, grad):
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        if self.dp_world_size > 1:
            grad = grad / self.dp_world_size
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                dist.all_reduce(grad, group=gpc.get_group(ParallelMode.DATA))
                ColoDDP._save_grad(p, grad)
            grad.record_stream(self.comm_stream)
        else:
            ColoDDP._save_grad(p, grad)
        return empty_grad

    @staticmethod
    def _save_grad(p, grad):
        if hasattr(p, '_saved_grad'):
            p._saved_grad.add_(grad)
        else:
            p._saved_grad = grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)
        for p in self.module.parameters():
            if getattr(p, '_saved_grad', None) is not None:
                if set_to_none:
                    p._saved_grad = None
                else:
                    if p._saved_grad.grad_fn is not None:
                        p._saved_grad.detach_()
                    else:
                        p._saved_grad.requires_grad_(False)
                    p._saved_grad.zero_()
