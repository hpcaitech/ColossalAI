import torch.nn

from colossalai.gemini.memory_tracer import MemStats
from colossalai.gemini.memory_tracer.model_data_memtracer import GLOBAL_CUDA_MEM_INFO
from colossalai.gemini.ophooks.runtime_mem_tracer_hook import GradMemTracerHook, ParamMemTracerHook
from colossalai.nn.parallel.data_parallel import _cast_float
from colossalai.tensor.param_op_hook import ColoParamOpHookManager

__all__ = ['RuntimeMemTracer']


class RuntimeMemTracer():
    """RuntimeMemTracer for the module training using ColoParameter.

    Trace non-model memory usage during fwd+bwd process.
    It is obtained by using a tensor with the same shape as the training process as the inputs
    and running an single fwd+bwd to trace the statistics.

    NOTE()
    1. The premise to use this tracer is that the target DNN execute the same operations at each iterations,
    2. Module buffers are viewed as non-model data.
    """

    def __init__(self, module: torch.nn.Module, dtype: torch.dtype = torch.half):
        super().__init__()
        self.module = module
        self.dtype = dtype
        self._memstats = MemStats()
        self.param_op_hook = ParamMemTracerHook(self._memstats)
        self.grad_hook = GradMemTracerHook(module)
        self.cpu_param_data_dict = {}

        for p in module.parameters():
            p.data = p.data.to(dtype)

        self._cast_buffers_to_cuda_dtype()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _backup_params(self):
        """
        The function is called before forward. Backup model params on cpu.
        """
        for p in self.module.parameters():
            self.cpu_param_data_dict[p] = torch.empty(p.data.shape, dtype=self.dtype, device="cpu")
            self.cpu_param_data_dict[p].copy_(p.data)

    def _restore_params(self):
        """
        This function is called after backward. Restore model params.
        """
        for p in self.module.parameters():
            p.data = torch.empty(p.data.shape, dtype=self.dtype, device="cpu", requires_grad=p.data.requires_grad)
            p.data.copy_(self.cpu_param_data_dict[p])
        self.cpu_param_data_dict.clear()

    def _pre_forward(self):
        self._clear_cuda_mem_info()
        self._backup_params()
        self.grad_hook.register_grad_hook()
        self.param_op_hook.mem_monitor.start()

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, self.dtype), _cast_float(kwargs, self.dtype)
        self.module.zero_grad(set_to_none=True)
        self._pre_forward()
        with ColoParamOpHookManager.use_hooks(self.param_op_hook):
            outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, loss):
        with self.param_op_hook.switch_to_backward(), ColoParamOpHookManager.use_hooks(self.param_op_hook):
            loss.backward()
        self._post_backward()

    def _post_backward(self):
        cuda_volume = self.param_op_hook.mem_monitor.finish()
        self._memstats.append_model_data('cuda', cuda_volume)
        self._memstats.append_non_model_data('cuda')
        # last_model_data = GLOBAL_CUDA_MEM_INFO.model_data_list[-1]
        # GLOBAL_CUDA_MEM_INFO.non_model_data_list.append(cuda_volume - last_model_data)
        self.grad_hook.remove_grad_hook()
        self._restore_params()

    def _clear_cuda_mem_info(self):
        # GLOBAL_CUDA_MEM_INFO.model_data_list.clear()
        # GLOBAL_CUDA_MEM_INFO.non_model_data_list.clear()
        self._memstats.clear()
        GLOBAL_CUDA_MEM_INFO.unreleased_grad_flag.clear()
        GLOBAL_CUDA_MEM_INFO.unreleased_grad_volume = 0

    def _cast_buffers_to_cuda_dtype(self):
        for buffer in self.module.buffers():
            buffer.data = buffer.cuda()
            if torch.is_floating_point(buffer):
                buffer.data = buffer.data.to(self.dtype)
