import os
import pathlib
import subprocess

from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def _build_cuda_native_kernel():

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    # Build path
    basepath = pathlib.Path(__file__).parent.absolute()
    srcpath = basepath / 'csrc'
    buildpath = basepath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                '-O3',
            ],
            extra_include_paths=[str(srcpath / 'kernels' / 'include')],
            extra_cuda_cflags=['-O3', '-gencode', 'arch=compute_70,code=sm_70', '--use_fast_math'] +
            extra_cuda_flags + cc_flag,
            verbose=False)

    # ==============
    # Fused softmax.
    # ==============

    extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda']
    
    # Upper triangular softmax.
    sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']
    colossal_scaled_upper_triang_masked_softmax = _cpp_extention_load_helper(
        "colossal_scaled_upper_triang_masked_softmax",
        sources, extra_cuda_flags)

    # Masked softmax.
    sources=[srcpath / 'scaled_masked_softmax.cpp',
                srcpath / 'scaled_masked_softmax_cuda.cu']
    colossal_scaled_masked_softmax = _cpp_extention_load_helper(
        "colossal_scaled_masked_softmax", sources, extra_cuda_flags)

    # =================================
    # Mixed precision fused layer norm.
    # =================================

    extra_cuda_flags = ['-maxrregcount=50']
    sources = [srcpath / 'layer_norm_cuda.cpp', srcpath / 'layer_norm_cuda_kernel.cu']
    colossal_layer_norm_cuda = _cpp_extention_load_helper("colossal_layer_norm_cuda", sources,
                                                          extra_cuda_flags)

    # ==========================================
    # Mixed precision Transformer Encoder Layer.
    # ==========================================

    extra_cuda_flags = ['-std=c++14',
                        '-U__CUDA_NO_HALF_OPERATORS__',
                        '-U__CUDA_NO_HALF_CONVERSIONS__',
                        '-U__CUDA_NO_HALF2_OPERATORS__',
                        '-DTHRUST_IGNORE_CUB_VERSION_CHECK']

    sources = [srcpath / 'multihead_attention_1d.cpp']
    kernel_sources = ["cublas_wrappers.cu",
                      "transform_kernels.cu",
                      "dropout_kernels.cu",
                      "normalize_kernels.cu",
                      "softmax_kernels.cu",
                      "general_kernels.cu",
                      "cuda_util.cu"]
    sources += [(srcpath / 'kernels' / cu_file) for cu_file in kernel_sources]
    colossal_multihead_attention = _cpp_extention_load_helper("colossal_multihead_attention", sources,
                                                          extra_cuda_flags)


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
