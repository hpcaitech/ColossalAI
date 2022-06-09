import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from functools import partial
from colossalai.zero.utils.zero_hook_v2 import ZeROHookV2
from colossalai.tensor.chunk import ChunkManager, TensorState, Chunk
from colossalai.tensor.param_op_hook import use_param_op_hooks
from colossalai.gemini.gemini_mgr import GeminiManager
from typing import Dict
from colossalai.logging import get_dist_logger


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
                    group = gpc.get_group(ParallelMode.DATA)
                    dist.all_reduce(grad, group=group)
                    ColoDDP._save_grad(p, grad)
                grad.record_stream(self.comm_stream)
            else:
                ColoDDP._save_grad(p, grad)
            return empty_grad

        else:
            group = gpc.get_cpu_group(ParallelMode.DATA)
            dist.all_reduce(grad, group=group)
            return grad

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


class ColoDDPV2(ColoDDP):

    def __init__(self, module: torch.nn.Module, gemini_manager: GeminiManager) -> None:
        super().__init__(module)
        self.gemini_manager = gemini_manager
        self.chunk_manager = gemini_manager.chunk_manager
        self.param_op_hook = ZeROHookV2(gemini_manager)
        self.fp32_params = []
        self.overflow_counter = 0
        self.grads_device: Dict[torch.Tensor, torch.device] = {}
        # TODO: get param order and filter unused params
        for p in module.parameters():
            assert p.dtype == torch.half
            fp32_p = p.float().detach()
            self.chunk_manager.append_tensor(p, 'fp16_param')
            self.chunk_manager.append_tensor(fp32_p, 'fp32_param')
            self.fp32_params.append(fp32_p)
            self.grads_device[p] = self.gemini_manager.default_device
        self._logger = get_dist_logger()

    def forward(self, *args, **kwargs):
        self.module.zero_grad(set_to_none=True)
        self.gemini_manager.pre_iter()
        with use_param_op_hooks(self.param_op_hook):
            outputs = self.module(*args, **kwargs)
        self.chunk_manager.exec_lazy_release()
        return outputs

    def _setup_grads_ptr(self):
        for p in self.module.parameters():
            if self.chunk_manager.get_chunk(p).is_free or not p.requires_grad:
                p.grad = None
            else:
                p.grad = p.data

    def _post_backward(self):
        self.chunk_manager.exec_lazy_release()
        self._setup_grads_ptr()
        self._logger.info(
            f'layout time: {self.gemini_manager._layout_time}, evict time: {self.gemini_manager._evict_time}, PCIE move vol: {self.gemini_manager._cpu_gpu_move_volume}B'
        )
        self.gemini_manager.post_iter()

    def backward(self, loss: torch.Tensor):
        with self.param_op_hook.switch_to_backward(), use_param_op_hooks(self.param_op_hook):
            loss.backward()
        self._post_backward()

    def backward_by_grad(self, tensor, grad):
        with self.param_op_hook.switch_to_backward(), use_param_op_hooks(self.param_op_hook):
            torch.autograd.backward(tensor, grad)
        self._post_backward()

    def grad_handle(self, p, grad):
        empty_grad = torch.empty_like(grad)
        free_storage(empty_grad)
        with torch._C.DisableTorchFunction():
            self.chunk_manager.trans_tensor_state(p, TensorState.READY_FOR_REDUCE)
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
            self.chunk_manager.copy_tensor_to_chunk_slice(p, grad)
            chunk = self.chunk_manager.get_chunk(p)
            reduced = self.chunk_manager.reduce_chunk(chunk)
            self.chunk_manager.release_chunk(chunk)
            if reduced and not chunk.is_free:
                self.overflow_counter += chunk.has_inf_or_nan
                self.chunk_manager.move_chunk(chunk, self.grads_device[p])
        return empty_grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.module.zero_grad(set_to_none=True)

    def _set_chunk_grad_device(self, chunk: Chunk, device: torch.device) -> None:
        for tensor in chunk.get_tensors():
            self.grads_device[tensor] = device
