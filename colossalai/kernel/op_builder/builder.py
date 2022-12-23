import os
import sys
from pathlib import Path


class Builder(object):

    def colossalai_src_path(self, code_path):
        if os.path.isabs(code_path):
            return code_path
        else:
            return os.path.join(Path(__file__).parent.parent.absolute(), code_path)

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
                         sources=self.strip_empty_entries(self.sources),
                         extra_include_paths=self.strip_empty_entries(self.extra_include_paths),
                         extra_cflags=self.extra_cxx_flags,
                         extra_cuda_cflags=self.extra_cuda_flags,
                         extra_ldflags=[],
                         verbose=verbose)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")

        return op_module
