try:
    from . import _meta_registrations
    META_COMPATIBILITY = True
except:
    import torch
    META_COMPATIBILITY = False

from .graph_module import ColoGraphModule
from .passes import MetaInfoProp
from .tracer import ColoTracer, meta_trace
