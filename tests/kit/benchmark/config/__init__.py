"""Layer 1, config: what is measured. ``model`` is what a model line declares once per model; ``plugins`` the plugins
and precisions to measure under; ``setup`` one measurement's conditions.

Only ``model`` is imported here: model_zoo imports this package, and ``plugins`` pulls in ColossalAI.
"""

from .model import DEFAULT_IMPLEMENTATION, BenchmarkSpec, causal_lm_batch, enable_checkpointing

__all__ = ["BenchmarkSpec", "DEFAULT_IMPLEMENTATION", "causal_lm_batch", "enable_checkpointing"]
