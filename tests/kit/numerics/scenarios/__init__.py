"""Scenarios: which runs to make and which rule judges them, one module per question.

- ``independent_impl``: the code under test against an independent implementation (``check_fwd_bwd``,
  ``check_training``).
"""

from . import independent_impl

__all__ = ["independent_impl"]
