import torch.nn

from colossalai.tensor.param_op_hook import ParamOpHookManager
from colossalai.gemini.ophooks import ParamMemHook
from colossalai.nn.parallel.data_parallel import _cast_float


class ParamWrapper():

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.param_op_hook = ParamMemHook()

        self._cast_buffers_to_cuda()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        self.param_op_hook.mem_monitor.start()

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, torch.float), _cast_float(kwargs, torch.float)
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

    def _cast_buffers_to_cuda(self):
        for buffer in self.module.buffers():
            buffer.data = buffer.cuda()
