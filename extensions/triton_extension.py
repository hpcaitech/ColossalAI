from .base_extension import _Extension

__all__ = ["_TritonExtension"]


class _TritonExtension(_Extension):
    def __init__(self, name: str, priority: int = 1):
        super().__init__(name, support_aot=False, support_jit=True, priority=priority)

    def is_hardware_compatible(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def load(self):
        return self.build_jit()
