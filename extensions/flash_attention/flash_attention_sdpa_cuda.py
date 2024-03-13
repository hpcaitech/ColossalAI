from ..base_extension import _Extension


class FlashAttentionSdpaCudaExtension(_Extension):
    def __init__(self):
        super().__init__(name="flash_attention_sdpa_cuda", support_aot=False, support_jit=False)

    def is_available(self) -> bool:
        # cuda extension can only be built if cuda is available
        try:
            import torch

            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        return cuda_available

    def assert_compatible(self) -> bool:
        pass

    def build_aot(self) -> None:
        raise NotImplementedError("Flash attention SDPA does not require ahead-of-time compilation.")

    def build_jit(self) -> None:
        raise NotImplementedError("Flash attention SDPA does not require just-in-time compilation.")

    def load(self):
        pass
