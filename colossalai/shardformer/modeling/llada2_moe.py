"""HF5 checkpoint metadata for the official LLaDA2.1/2.2 MoE models."""

import sys
from functools import lru_cache

__all__ = ["get_llada2_model_class"]


@lru_cache(maxsize=None)
def get_llada2_model_class(original_class):
    """Adapt an already trusted LLaDA2MoeModelLM class for HF5 save/load.

    The MoE, attention and same-position loss implementations are inherited.
    Save/load uses the returned class again, not AutoModel's original class.
    """
    module = sys.modules[original_class.__module__]
    if original_class is not getattr(module, "LLaDA2MoeModelLM", None):
        raise TypeError("Pass the official remote-code LLaDA2MoeModelLM class")

    class LLaDA2MoeModelLM(original_class):
        _auto_class = None
        _tied_weights_keys = {"lm_head.weight": "model.word_embeddings.weight"}

        def __init__(self, config):
            from transformers import __version__

            if __version__ != "5.17.0":
                raise NotImplementedError("LLaDA2 compatibility is validated on transformers==5.17.0")
            super().__init__(config)

    LLaDA2MoeModelLM.__qualname__ = "LLaDA2MoeModelLM"
    return LLaDA2MoeModelLM
