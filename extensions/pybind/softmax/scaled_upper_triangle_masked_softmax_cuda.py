from ...cuda_extension import _CudaExtension
from ...utils import append_nvcc_threads, get_cuda_cc_flag


class ScaledUpperTriangleMaskedSoftmaxCudaExtension(_CudaExtension):
    def __init__(self):
        super().__init__(name="scaled_upper_triangle_masked_softmax_cuda")

    def sources_files(self):
        ret = [
            self.csrc_abs_path(fname)
            for fname in [
                "kernel/cuda/scaled_upper_triang_masked_softmax_kernel.cu",
            ]
        ] + [self.pybind_abs_path("softmax/scaled_upper_triang_masked_softmax.cpp")]
        return ret

    def cxx_flags(self):
        return ["-O3"] + self.version_dependent_macros

    def nvcc_flags(self):
        extra_cuda_flags = [
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]
        extra_cuda_flags.extend(get_cuda_cc_flag())
        ret = ["-O3", "--use_fast_math"] + extra_cuda_flags + super().nvcc_flags()
        return append_nvcc_threads(ret)
