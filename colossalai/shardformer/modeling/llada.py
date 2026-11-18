"""HF5 loading metadata for the official LLaDA implementation."""

import sys
from functools import lru_cache

__all__ = ["get_llada_model_class"]


@lru_cache(maxsize=None)
def get_llada_model_class(original_class):
    """Adapt an already trusted LLaDAModelLM class without replacing its math.

    Use Class(config, init_params=True) for random initialization, or
    Class.from_pretrained(path, config=config) to load weights. LLaDA's own
    constructor owns initialization; HF post_init must not initialize it twice.
    Save/load uses this class again, rather than AutoModel's original class.
    """
    module = sys.modules[original_class.__module__]
    if original_class is not getattr(module, "LLaDAModelLM", None):
        raise TypeError("Pass the official remote-code LLaDAModelLM class")

    class LLaDAModelLM(original_class):
        _auto_class = None
        _tied_weights_keys = {}

        def __init__(self, config, model=None, init_params=False):
            from transformers import __version__

            if __version__ != "5.17.0":
                raise NotImplementedError("LLaDA compatibility is validated on transformers==5.17.0")
            super().__init__(config, model=model, init_params=init_params)
            self.post_init()

        def init_weights(self):
            # Official LLaDAModel(..., init_params=True) already initializes
            # weights. post_init is needed only for HF5's loading metadata.
            self.tie_weights(recompute_mapping=False)

        def tie_weights(self, missing_keys=None, recompute_mapping=True):
            # With weight_tying=True the official forward directly uses wte
            # as its output projection: there is only ONE state_dict entry.
            # Its old tie_weights creates an unused ff_out alias and does not
            # accept HF5's keyword arguments. No alias needs creating here.
            return None

    LLaDAModelLM.__qualname__ = "LLaDAModelLM"
    return LLaDAModelLM
