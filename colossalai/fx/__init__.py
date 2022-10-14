try:
    from . import _meta_registrations
    META_COMPATIBILITY = True
except:
    import torch
    META_COMPATIBILITY = False
    print(f'_meta_registrations seems to be incompatible with PyTorch {torch.__version__}.')
from .tracer import ColoTracer, meta_trace
from .graph_module import ColoGraphModule
from .passes import MetaInfoProp
