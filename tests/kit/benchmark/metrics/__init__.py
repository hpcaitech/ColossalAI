"""Layer 3, metrics: pure functions from the runner's raw records to metrics. Each metric is computed here and nowhere
else, so a single measurement, a search, a scaling curve and a comparison report the same numbers the same way.

Nothing is imported here: model_zoo imports ``compute`` for its FLOPs formulas, and must not pay for the rest.
"""
