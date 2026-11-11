"""Benchmark framework: throughput and peak memory of model_zoo entries at their real sizes, measured the same way for
every model and written as one JSON report format with one Markdown template. See README.md.

Only ``spec`` and ``flops`` are imported here: model_zoo entries import them, and they must not pull in ColossalAI.
"""

from .flops import decoder_flops_per_token
from .spec import BenchmarkSpec, causal_lm_batch

__all__ = ["BenchmarkSpec", "causal_lm_batch", "decoder_flops_per_token"]
