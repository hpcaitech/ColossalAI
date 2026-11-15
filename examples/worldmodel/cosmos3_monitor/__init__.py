"""Read-only monitoring for the Cosmos 3 SFT and replay training entries.

The package turns the JSONL telemetry that :mod:`examples.worldmodel.cosmos3_harness.training_metrics`
already writes into a durable run registry, a small HTTP API and an offline single-page dashboard.  The
files a training process produces are the source of truth, and every view here can be rebuilt from them.

The one write path is the control channel (:mod:`.control`): a stop or resume request dropped into
``<run>/control/request.json``, which the training loop's watcher thread (``cosmos3_harness.run_control``)
honours after the current step.  ``serve --no-control`` turns it off and leaves the dashboard read-only.
"""

from __future__ import annotations

__all__ = ["MONITOR_VERSION", "SCHEMA_VERSION"]

#: Schema of the monitor's own optional files (``checkpoints.jsonl``, ``system/``, ...).  Bumped only when a
#: field changes meaning, never when one is added.
SCHEMA_VERSION = 1

#: Version of the dashboard package, shown in ``/api/health``.
MONITOR_VERSION = "0.1.0"
