r"""Qwen3-MoE sharding helpers, selected by the installed transformers major version.

Only ``_qwen3_moe_tf5`` exists for transformers v5; ``_qwen3_moe_tf4`` is a stub that fails with a clear message. See
:func:`colossalai._compat.reexport_transformers_impl`.
"""

from colossalai._compat import reexport_transformers_impl

reexport_transformers_impl(globals(), __package__, "_qwen3_moe")
