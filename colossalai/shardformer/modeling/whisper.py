from colossalai._compat import reexport_transformers_impl

reexport_transformers_impl(globals(), __package__, "_whisper")
