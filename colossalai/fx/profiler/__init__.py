try:
    from ._meta_registrations import *
except:
    import torch
    print(f'_meta_registrations seems to be incompatible with PyTorch {torch.__version__}.')
from .meta_tensor import MetaTensor
from .registry import meta_profiler_function, meta_profiler_module
from .profiler_function import *
from .profiler_module import *
from .profiler import *
