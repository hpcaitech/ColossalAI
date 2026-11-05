"""Scenarios: which runs to make and which rule judges them, one module per question.

- ``independent_impl``: the code under test against an independent implementation (``check_fwd_bwd``,
  ``check_training``).
- ``data_parallel``, ``resume``: two runs of the same code that differ in one condition (``check``).
"""

from . import data_parallel, independent_impl, resume

__all__ = ["data_parallel", "independent_impl", "resume"]
