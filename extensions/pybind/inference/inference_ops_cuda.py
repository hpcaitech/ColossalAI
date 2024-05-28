from ...cuda_extension import _CudaExtension
from ...utils import get_cuda_cc_flag


class InferenceOpsCudaExtension(_CudaExtension):
    def __init__(self):
        super().__init__(name="inference_ops_cuda")

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                "kernel/cuda/decode_kv_cache_memcpy_kernel.cu",
                "kernel/cuda/context_kv_cache_memcpy_kernel.cu",
                "kernel/cuda/fused_rotary_emb_and_cache_kernel.cu",
                "kernel/cuda/activation_kernel.cu",
                "kernel/cuda/rms_layernorm_kernel.cu",
                "kernel/cuda/get_cos_and_sin_kernel.cu",
                "kernel/cuda/flash_decoding_attention_kernel.cu",
                "kernel/cuda/convert_fp8_kernel.cu",
            ]
        ] + [self.pybind_abs_path("inference/inference.cpp")]
        return ret

    def cxx_flags(self):
        version_dependent_macros = ["-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5"]
        return ["-O3"] + version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = ["-lineinfo"]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        return ["-O3", "--use_fast_math"] + extra_cuda_flags + super().nvcc_flags()
