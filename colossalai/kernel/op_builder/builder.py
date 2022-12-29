import os
import re
import sys
from pathlib import Path

import torch


def get_cuda_cc_flag():
    """get_cuda_cc_flag

    cc flag for your GPU arch
    """
    cc_flag = []
    for arch in torch.cuda.get_arch_list():
        res = re.search(r'sm_(\d+)', arch)
        if res:
            arch_cap = res[1]
            if int(arch_cap) >= 60:
                cc_flag.extend(['-gencode', f'arch=compute_{arch_cap},code={arch}'])

    return cc_flag


class Builder(object):

    def colossalai_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

    def get_cuda_home_include(self):
        """
        return include path inside the cuda home.
        """
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None:
            raise RuntimeError("CUDA_HOME is None, please set CUDA_HOME to compile C++/CUDA kernels in ColossalAI.")
        cuda_include = os.path.join(CUDA_HOME, "include")
        return cuda_include

    # functions must be overrided begin
    def sources_files(self):
        raise NotImplementedError

    def include_dirs(self):
        raise NotImplementedError

    def cxx_flags(self):
        raise NotImplementedError

    def nvcc_flags(self):
        raise NotImplementedError

    # functions must be overrided over

    def strip_empty_entries(self, args):
        '''
        Drop any empty strings from the list of compile and link flags
        '''
        return [x for x in args if len(x) > 0]

    def load(self, verbose=True):
        """

        load and compile cpu_adam lib at runtime

        Args:
            verbose (bool, optional): show detailed info. Defaults to True.
        """
        import time

        from torch.utils.cpp_extension import load
        start_build = time.time()

        op_module = load(name=self.name,
                         sources=self.strip_empty_entries(self.sources_files()),
                         extra_include_paths=self.strip_empty_entries(self.include_dirs()),
                         extra_cflags=self.cxx_flags(),
                         extra_cuda_cflags=self.nvcc_flags(),
                         extra_ldflags=[],
                         verbose=verbose)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        return op_module

    def builder(self, name) -> 'CUDAExtension':
        """
        get a CUDAExtension instance used for setup.py
        """
        from torch.utils.cpp_extension import CUDAExtension

        return CUDAExtension(
            name=name,
            sources=[os.path.join('colossalai/kernel/cuda_native/csrc', path) for path in self.sources_files()],
            include_dirs=self.include_dirs(),
            extra_compile_args={
                'cxx': self.cxx_flags(),
                'nvcc': self.nvcc_flags()
            })
