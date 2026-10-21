r"""Llama sharding policies, selected by the installed transformers major version.

``_llama_tf5`` targets transformers v5; ``_llama_tf4`` is the implementation for the pinned v4
(``transformers==4.51.3``). See :func:`colossalai._compat.reexport_transformers_impl`.
"""

from colossalai._compat import reexport_transformers_impl

reexport_transformers_impl(globals(), __package__, "_llama")
