import torch
import torch.nn as nn

class GradOffloadHook():

    def __init__(self):
        self.grad_hook_list = []

    def grad_handle(self, grad):
        grad.data = grad.data.to("cpu")
        return grad

    def register_grad_hook(self, module: torch.nn.Module):
        for p in module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(self.grad_handle))

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()

class OffloadModuleWrapper:

    def __init__(self, model: nn.Module):
        self.model = model
        self.grad_offload_hook = GradOffloadHook()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        self.grad_offload_hook.register_grad_hook(self.model)

    def forward(self, *args, **kwargs):
        self.model.zero_grad(set_to_none=True)
        self._pre_forward()
        outputs = self.model(*args, **kwargs)
        return outputs

    def backward(self, loss):
        loss.backward()
        self._post_backward()

    def _post_backward(self):
        self.grad_offload_hook.remove_grad_hook()
