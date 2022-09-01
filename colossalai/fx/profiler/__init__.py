try:
    from ._meta_registrations import *
except:
    import torch
    print(f'_meta_registrations seems to be incompatible with PyTorch {torch.__version__}.')
from .tensor import MetaTensor
from .memory import parameter_size, activation_size
from .profiler import profile_function, profile_method, profile_module, _profile
