import os
import re

from setuptools import Extension, find_packages, setup

from colossalai.kernel.op_builder.utils import get_cuda_bare_metal_version

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])

    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 10):
        raise RuntimeError("Colossal-AI requires Pytorch 1.10 or newer.\n"
                           "The latest stable release can be obtained from https://pytorch.org/")
except ImportError:
    raise ModuleNotFoundError('torch is not found. You need to install PyTorch before installing Colossal-AI.')

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
build_cuda_ext = True
ext_modules = []

if int(os.environ.get('NO_CUDA_EXT', '0')) == 1:
    build_cuda_ext = False


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_major = torch.version.cuda.split(".")[0]
    torch_binary_minor = torch.version.cuda.split(".")[1]

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if bare_metal_major != torch_binary_major:
        print(f'The detected CUDA version ({raw_output}) mismatches the version that was used to compile PyTorch '
              f'({torch.version.cuda}). CUDA extension will not be installed.')
        return False

    if bare_metal_minor != torch_binary_minor:
        print("\nWarning: Cuda extensions are being compiled with a version of Cuda that does "
              "not match the version used to compile Pytorch binaries.  "
              f"Pytorch binaries were compiled with Cuda {torch.version.cuda}.\n"
              "In some cases, a minor-version mismatch will not cause later errors:  "
              "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798. ")
    return True


def check_cuda_availability(cuda_dir):
    if not torch.cuda.is_available():
        # https://github.com/NVIDIA/apex/issues/486
        # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query
        # torch.cuda.get_device_capability(), which will fail if you are compiling in an environment
        # without visible GPUs (e.g. during an nvidia-docker build command).
        print(
            '\nWarning: Torch did not find available GPUs on this system.\n',
            'If your intention is to cross-compile, this is not an error.\n'
            'By default, Colossal-AI will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
            'Volta (compute capability 7.0), Turing (compute capability 7.5),\n'
            'and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n'
            'If you wish to cross-compile for a single specific architecture,\n'
            'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
        if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
            _, bare_metal_major, _ = get_cuda_bare_metal_version(cuda_dir)
            if int(bare_metal_major) == 11:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
        return False

    if cuda_dir is None:
        print("nvcc was not found. CUDA extension will not be installed. If you're installing within a container from "
              "https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.")
        return False
    return True


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


def fetch_requirements(path):
    with open(path, 'r') as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version():
    setup_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(setup_file_path)
    version_txt_path = os.path.join(project_path, 'version.txt')
    version_py_path = os.path.join(project_path, 'colossalai/version.py')

    with open(version_txt_path) as f:
        version = f.read().strip()
        if build_cuda_ext:
            torch_version = '.'.join(torch.__version__.split('.')[:2])
            cuda_version = '.'.join(get_cuda_bare_metal_version(CUDA_HOME)[1:])
            version += f'+torch{torch_version}cu{cuda_version}'

    # write version into version.py
    with open(version_py_path, 'w') as f:
        f.write(f"__version__ = '{version}'\n")

    return version


if build_cuda_ext:
    build_cuda_ext = check_cuda_availability(CUDA_HOME) and check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

if build_cuda_ext:
    # Set up macros for forward/backward compatibility hack around
    # https://github.com/pytorch/pytorch/commit/4404762d7dd955383acee92e6f06b48144a0742e
    # and
    # https://github.com/NVIDIA/apex/issues/456
    # https://github.com/pytorch/pytorch/commit/eb7b39e02f7d75c26d8a795ea8c7fd911334da7e#diff-4632522f237f1e4e728cb824300403ac
    version_dependent_macros = ['-DVERSION_GE_1_1', '-DVERSION_GE_1_3', '-DVERSION_GE_1_5']

    def cuda_ext_helper(name, sources, extra_cuda_flags, extra_cxx_flags=[]):
        return CUDAExtension(
            name=name,
            sources=[os.path.join('colossalai/kernel/cuda_native/csrc', path) for path in sources],
            include_dirs=[os.path.join(this_dir, 'colossalai/kernel/cuda_native/csrc/kernels/include')],
            extra_compile_args={
                'cxx': ['-O3'] + version_dependent_macros + extra_cxx_flags,
                'nvcc': append_nvcc_threads(['-O3', '--use_fast_math'] + version_dependent_macros + extra_cuda_flags)
            })

    #### fused optim kernels ###
    from colossalai.kernel.op_builder import FusedOptimBuilder
    ext_modules.append(FusedOptimBuilder().builder('colossalai._C.fused_optim'))

    #### N-D parallel kernels ###
    cc_flag = []
    for arch in torch.cuda.get_arch_list():
        res = re.search(r'sm_(\d+)', arch)
        if res:
            arch_cap = res[1]
            if int(arch_cap) >= 60:
                cc_flag.extend(['-gencode', f'arch=compute_{arch_cap},code={arch}'])

    extra_cuda_flags = [
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '--expt-relaxed-constexpr',
        '--expt-extended-lambda'
    ]

    from colossalai.kernel.op_builder import ScaledSoftmaxBuilder
    ext_modules.append(ScaledSoftmaxBuilder().builder('colossalai._C.scaled_upper_triang_masked_softmax'))

    ext_modules.append(
        cuda_ext_helper('colossalai._C.scaled_masked_softmax',
                        ['scaled_masked_softmax.cpp', 'scaled_masked_softmax_cuda.cu'], extra_cuda_flags + cc_flag))

    ext_modules.append(
        cuda_ext_helper('colossalai._C.moe', ['moe_cuda.cpp', 'moe_cuda_kernel.cu'], extra_cuda_flags + cc_flag))

    extra_cuda_flags = ['-maxrregcount=50']

    ext_modules.append(
        cuda_ext_helper('colossalai._C.layer_norm', ['layer_norm_cuda.cpp', 'layer_norm_cuda_kernel.cu'],
                        extra_cuda_flags + cc_flag))

    ### MultiHeadAttn Kernel ####
    from colossalai.kernel.op_builder import MultiHeadAttnBuilder
    ext_modules.append(MultiHeadAttnBuilder().builder('colossalai._C.multihead_attention'))

    ### Gemini Adam kernel ####
    from colossalai.kernel.op_builder import CPUAdamBuilder
    ext_modules.append(CPUAdamBuilder().builder('colossalai._C.cpu_optim'))

setup(name='colossalai',
      version=get_version(),
      packages=find_packages(exclude=(
          'benchmark',
          'docker',
          'tests',
          'docs',
          'examples',
          'tests',
          'scripts',
          'requirements',
          '*.egg-info',
      )),
      description='An integrated large-scale model training system with efficient parallelization techniques',
      long_description=fetch_readme(),
      long_description_content_type='text/markdown',
      license='Apache Software License 2.0',
      url='https://www.colossalai.org',
      project_urls={
          'Forum': 'https://github.com/hpcaitech/ColossalAI/discussions',
          'Bug Tracker': 'https://github.com/hpcaitech/ColossalAI/issues',
          'Examples': 'https://github.com/hpcaitech/ColossalAI-Examples',
          'Documentation': 'http://colossalai.readthedocs.io',
          'Github': 'https://github.com/hpcaitech/ColossalAI',
      },
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension} if ext_modules else {},
      install_requires=fetch_requirements('requirements/requirements.txt'),
      entry_points='''
        [console_scripts]
        colossalai=colossalai.cli:cli
    ''',
      python_requires='>=3.6',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: Apache Software License',
          'Environment :: GPU :: NVIDIA CUDA',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: System :: Distributed Computing',
      ],
      package_data={'colossalai': ['_C/*.pyi']})
