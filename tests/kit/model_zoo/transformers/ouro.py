"""Offline Ouro model-zoo registrations, with no import-time remote-code loading."""

from functools import partial

from transformers import __version__

from tests.kit.ouro import make_ouro, ouro_entry

from ..registry import ModelAttribute, model_zoo

if __version__ == "5.17.0":
    for _key in ("ouro14", "ouro26"):
        _entry = ouro_entry(_key)
        model_zoo.register(
            f"transformers_{_key}_for_causal_lm",
            partial(make_ouro, _key, apply_policy=False),
            *_entry[1:4],
            model_attribute=ModelAttribute(has_control_flow=True, numerics=_entry[4].numerics),
        )
