"""Layer 4, scenarios: which setups to run and how to report them. Each scenario changes one field of a setup and
measures again: ``single`` (one setup), ``search`` (the sequence length or micro-batch), ``scaling`` (the device
count), ``compare`` (any one field, e.g. the plugin or the implementation). ``report`` writes the JSON and the
Markdown. Nothing here computes a metric."""
