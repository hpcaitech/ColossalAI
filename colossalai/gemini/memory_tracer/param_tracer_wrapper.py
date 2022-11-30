import torch.nn

from colossalai.tensor.param_op_hook import ParamOpHookManager
from colossalai.gemini.ophooks.param_trace_hook import ParamTracerHook
from colossalai.gemini.tensor_utils import free_storage
from colossalai.nn.parallel.data_parallel import _cast_float
from functools import partial


__all__ = ['ParamTracerWrapper']

class ParamTracerWrapper():

    def __init__(self, module: torch.nn.Module, dtype: torch.dtype = torch.half):
        super().__init__()
        self.module = module
        self.dtype = dtype
        self.param_op_hook = ParamTracerHook(dtype)

        for p in module.parameters():
            p.data = p.data.to(dtype)
            if p.requires_grad:
                p.register_hook(partial(self.grad_handle))

        self._cast_buffers_to_cuda_dtype()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def grad_handle(self, grad):
        free_storage(grad)

    def _pre_forward(self):
        self.param_op_hook.mem_monitor.start()

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, self.dtype), _cast_float(kwargs, self.dtype)
        self.module.zero_grad(set_to_none=True)
        self._pre_forward()
        with ParamOpHookManager.use_hooks(self.param_op_hook):
            outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, loss):
        with self.param_op_hook.switch_to_backward(), ParamOpHookManager.use_hooks(self.param_op_hook):
            loss.backward()
        self._post_backward()

    def _post_backward(self):
        cuda_volume = self.param_op_hook.mem_monitor.finish()
        last_model_data = self.param_op_hook._model_data_list[-1]
        self.param_op_hook._non_model_data_list.append(cuda_volume - last_model_data)

    def _cast_buffers_to_cuda_dtype(self):
        for buffer in self.module.buffers():
            buffer.data = buffer.cuda()
            if torch.is_floating_point(buffer):
                buffer.data = buffer.data.to(self.dtype)