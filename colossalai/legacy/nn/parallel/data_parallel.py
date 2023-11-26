from collections import OrderedDict
from functools import partial
from typing import Iterable, Optional, Set

import torch
import torch.distributed as dist

from colossalai.legacy.tensor import ProcessGroup as ColoProcessGroup
from colossalai.utils import is_ddp_ignored

from .reducer import Reducer


def free_storage(data: torch.Tensor) -> None:
    """Free underlying storage of a Tensor."""
    if data.storage().size() > 0:
        # Since we're modifying the Tensor's Storage directly, make sure the Tensor
        # is the sole occupant of the Storage.
        assert data.storage_offset() == 0
        data.storage().resize_(0)


def _cast_float(args, dtype: torch.dtype):
    if isinstance(args, torch.Tensor) and torch.is_floating_point(args):
        args = args.to(dtype)
    elif isinstance(args, (list, tuple)):
        args = type(args)(_cast_float(t, dtype) for t in args)
    elif isinstance(args, dict):
        args = {k: _cast_float(v, dtype) for k, v in args.items()}
    return args


class ColoDDP(torch.nn.Module):
    """Distributed data parallel for ColoTensor. Nested ColoDDP is not supported now.

    Example:
        >>> from colossalai.legacy.core import global_context as gpc
        >>> from colossalai.legacy.context import ParallelMode
        >>> model = torch.nn.Linear(20, 1)
        >>> pg = ProcessGroup(tp_degree = world_size//2)
        >>> model = ColoDDP(model, pg)
        >>> logits = model(x)
        >>> loss = criterion(logits, labels)
        >>> model.backward(loss)

    Args:
        module (torch.nn.Module): Module to apply DDP.
        process_group (Optional[dist.ProcessGroup], optional): The process group which DDP uses.
            If it's None, the default data parallel group will be used. Defaults to None.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        process_group: ColoProcessGroup,
        bucket_cap_mb: int = 25,
        rebuild_bucket: bool = True,
    ) -> None:
        assert not isinstance(module, ColoDDP)
        super().__init__()
        self.module = module
        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        assert process_group

        self.process_group = process_group
        self.dp_world_size = self.process_group.dp_world_size()

        self.reducer = Reducer(bucket_cap_mb)
        self.rebuild_bucket = rebuild_bucket
        for p in module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle, p))

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        return self.module.named_buffers(prefix, recurse)

    def named_children(self):
        return self.module.named_children()

    def named_modules(
        self, memo: Optional[Set[torch.nn.Module]] = None, prefix: str = "", remove_duplicate: bool = True
    ):
        return self.module.named_modules(memo, prefix, remove_duplicate)

    def forward(self, *args, **kwargs):
        self.module.zero_grad(set_to_none=True)
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        loss.backward()
        with torch.cuda.stream(self.comm_stream):
            self.reducer.flush()
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        if self.rebuild_bucket:
            self.reducer.free()
        for p in self.module.parameters():
            if is_ddp_ignored(p):
                continue
            if p.grad.device.type != "cpu":
                p.grad = p._saved_grad

    def grad_handle(self, p, grad):
        if grad.device.type != "cpu":
            empty_grad = torch.empty_like(grad)
            free_storage(empty_grad)
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    self.reducer.all_reduce_async(
                        grad, group=self.process_group.dp_process_group(), callback_fn=partial(self._save_grad, p)
                    )
                grad.record_stream(self.comm_stream)
            else:
                ColoDDP._save_grad(p, grad)
            return empty_grad

        else:
            # TODO(jiaruifang) fixme
            self.process_group.set_cpu_groups()
            dist.all_reduce(grad, group=self.process_group.cpu_dp_process_group())
            return grad

    @staticmethod
    def _save_grad(p, grad):
        if hasattr(p, "_saved_grad"):
            p._saved_grad.add_(grad)
        else:
            p._saved_grad = grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)
        for p in self.module.parameters():
            if getattr(p, "_saved_grad", None) is not None:
                if set_to_none:
                    p._saved_grad = None
                else:
                    if p._saved_grad.grad_fn is not None:
                        p._saved_grad.detach_()
                    else:
                        p._saved_grad.requires_grad_(False)
                    p._saved_grad.zero_()

    @staticmethod
    def set_params_to_ignore(params_to_ignore: Iterable[torch.Tensor]) -> None:
        """Sets parameters to be ignored by DDP.
        This method must be called before initializing ColoDDP.

        Example:
            >>> params_to_ignore = []
            >>> for p in module.parameters():
            >>>     if should_ignore(p):
            >>>         params_to_ignore.append(p)
            >>> ColoDDP.set_params_to_ignore(params_to_ignore)
            >>> module = ColoDDP(module)

        Args:
            params_to_ignore (Iterable[torch.Tensor]): A list of parameters to be ignored.
        """
        for p in params_to_ignore:
            p._ddp_to_ignore = True

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]", strict: bool = True):
        return self.module.load_state_dict(state_dict, strict)
