"""Benchmark framework: throughput, MFU and memory of model_zoo entries at their real sizes, measured the same way for
every model and written as one JSON report format with one Markdown template. See README.md.

Four layers, one package each; a layer only uses the ones below it:

- ``config``: what is measured (a model's ``BenchmarkSpec``, the plugin configurations, one ``Setup``);
- ``runner``: runs one setup once and records raw data, computing no metric;
- ``metrics``: turns raw data into metrics, pure functions, each metric computed in one place;
- ``scenarios``: decides which setups to run (single, maximum search, scaling, comparison) and writes the report.

Only what model_zoo entries need is imported here, and none of it pulls in ColossalAI.
"""

from .config.model import BenchmarkSpec, causal_lm_batch
from .metrics.compute import decoder_flops_per_token

__all__ = ["BenchmarkSpec", "causal_lm_batch", "decoder_flops_per_token"]
