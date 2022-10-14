try:
    from . import _meta_registrations
    META_COMPATIBILITY = True
except:
    import torch
    META_COMPATIBILITY = False
    print(f'_meta_registrations seems to be incompatible with PyTorch {torch.__version__}.')
from .initialize import (initialize, launch, launch_from_openmpi, launch_from_slurm, launch_from_torch,
                         get_default_parser)

__version__ = '0.1.10'
