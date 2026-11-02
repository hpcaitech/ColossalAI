"""Layer 1, capture: runs that each produce one ``Trace``.

``fwd_bwd`` (one forward and backward of a function or module) and ``utils`` depend only on torch and are exported
here. ``recipes`` and ``training`` (training steps through a booster plugin) depend on ColossalAI and are not
imported here; import them as ``tests.kit.numerics.capture.training``.
"""

from .fwd_bwd import run_fwd_bwd
from .utils import cast_floating, full_precision_matmul, missing_grads, upstream_loss

__all__ = ["cast_floating", "full_precision_matmul", "missing_grads", "run_fwd_bwd", "upstream_loss"]
