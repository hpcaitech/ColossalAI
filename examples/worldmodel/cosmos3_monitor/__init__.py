"""Read-only monitoring for the Cosmos 3 SFT and replay training entries.

The package turns the JSONL telemetry that :mod:`examples.worldmodel.cosmos3_harness.training_metrics`
already writes into a durable run registry, a small read-only HTTP API and an offline single-page
dashboard.  It never writes inside a run directory: the files a training process produces are the source
of truth, and every view here can be rebuilt from them.
"""

from __future__ import annotations

__all__ = ["MONITOR_VERSION", "SCHEMA_VERSION"]

#: Schema of the monitor's own optional files (``checkpoints.jsonl``, ``system/``, ...).  Bumped only when a
#: field changes meaning, never when one is added.
SCHEMA_VERSION = 1

#: Version of the dashboard package, shown in ``/api/health``.
MONITOR_VERSION = "0.1.0"
