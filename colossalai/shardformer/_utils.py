import re


def is_hf5():
    from packaging.version import Version
    from transformers import __version__

    return Version(__version__).major >= 5


def hf5_policy_error(family, shard_config=None):
    """Return a declared HF5 support-boundary error, not a compatibility guess."""
    if not is_hf5():
        return None
    reasons = {
        "chatglm2": "remote-code revision and HF5 compatibility have not been validated",
        "deepseek": "legacy remote-code policy is not validated for HF5 native Deepseek models",
    }
    if family in reasons:
        reason = reasons[family]
    elif family in {"opt", "command", "blip2"} and shard_config is not None:
        from packaging.version import Version
        from transformers import __version__

        if Version(__version__) < Version("5.17.0"):
            return (
                f"ShardFormer {family} HF5 compatibility requires transformers>=5.17.0; installed {__version__}. "
                "See docs/HF5-COMPATIBILITY.md."
            )
        unsupported = [
            name
            for name in (
                "enable_tensor_parallelism",
                "enable_sequence_parallelism",
                "enable_flash_attention",
                "enable_jit_fused",
                "enable_fused_normalization",
                "enable_all_optimization",
                "fp8_communication",
            )
            if getattr(shard_config, name, False)
        ]
        if getattr(shard_config, "pipeline_stage_manager", None) is not None:
            unsupported.append("pipeline_parallelism")
        if not unsupported:
            return None
        reason = (
            "HF5 migration is currently validated only without parallel/optional optimization; requested "
            + ", ".join(unsupported)
        )
    else:
        return None
    from transformers import __version__

    return (
        f"ShardFormer {family} policy does not support this configuration with transformers {__version__}: "
        f"{reason}. See docs/HF5-COMPATIBILITY.md. Use the documented validated configuration, "
        "or a separate legacy environment following the repository's transformers==4.51.3 declaration "
        "(not a guarantee that every model/configuration is validated)."
    )


def check_hf5_policy(family, shard_config=None):
    error = hf5_policy_error(family, shard_config)
    if error is not None:
        raise NotImplementedError(error)


def get_hf_mask_config(config, *, eager=False):
    """Select an explicit mask backend without mutating the model's shared config."""
    from copy import copy

    if not eager:
        return config
    config = copy(config)
    config._attn_implementation = "eager"
    return config


def prepare_hf_bidirectional_mask(config, inputs_embeds, attention_mask=None, encoder_hidden_states=None):
    """Build an explicit additive mask for vision or encoder cross-attention."""
    from transformers.masking_utils import create_bidirectional_mask

    return create_bidirectional_mask(
        config=get_hf_mask_config(config, eager=True),
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
    )


def prepare_hf_encoder_decoder_cache(config, past_key_values=None, *, encoder_hidden_states=None):
    """Allocate decoder-sized self/cross caches, preserving an existing Cache object.

    Whisper maps num_hidden_layers to encoder_layers. Override it on a private
    copy so distilled models allocate only the actual number of decoder slots.
    """
    from copy import deepcopy

    from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache

    if past_key_values is not None and not isinstance(past_key_values, Cache):
        raise TypeError("HF 5.17 requires a Cache object, not legacy tuple past_key_values")
    needs_cross = encoder_hidden_states is not None or config.is_encoder_decoder
    if needs_cross and isinstance(past_key_values, DynamicCache):
        if past_key_values.get_seq_length() != 0:
            raise ValueError("Use EncoderDecoderCache for nonempty encoder-decoder past_key_values")
        past_key_values = None
    if past_key_values is None:
        cache_config = deepcopy(config)
        cache_config.num_hidden_layers = config.decoder_layers
        past_key_values = DynamicCache(config=cache_config)
        if needs_cross:
            past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=cache_config))
    return past_key_values


def get_hf_cache_seq_length(past_key_values, layer_idx: int = 0):
    """Read the number of seen tokens using the HF Cache contract, including an empty cache."""
    if past_key_values is None:
        return 0
    return past_key_values.get_seq_length(layer_idx)


def prepare_hf_causal_masks(config, inputs_embeds, attention_mask, past_key_values, position_ids, *, eager=False):
    """Build HF 5 masks by layer type without changing the model's backend config.

    ShardFormer's eager attention and ColoAttention need explicit additive masks;
    an SDPA mask may otherwise be None (implicit causal) or boolean. Cache-aware
    HF mask creation also handles sliding-window offsets and precomputed masks.
    Imports stay local so unrelated models do not require this HF 5 API.
    """
    from copy import copy

    import torch
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    if eager:
        config = copy(config)
        config._attn_implementation = "eager"
    kwargs = dict(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    factories = {"full_attention": create_causal_mask}
    if "sliding_attention" in config.layer_types:
        factories["sliding_attention"] = create_sliding_window_causal_mask
    masks = {}
    for layer_type, factory in factories.items():
        if isinstance(attention_mask, dict):
            mask = attention_mask[layer_type]
            # SDPA's None means implicit causality, not unrestricted attention.
            if mask is None and eager:
                mask = factory(**{**kwargs, "attention_mask": None})
        else:
            mask = factory(**kwargs)
        if eager and mask is not None and mask.dtype == torch.bool:
            additive_mask = torch.zeros_like(mask, dtype=inputs_embeds.dtype)
            mask = additive_mask.masked_fill(~mask, torch.finfo(inputs_embeds.dtype).min)
        masks[layer_type] = mask
    return masks


def get_hf_flash_attention_kwargs(attention_mask):
    """Pass an explicit HF additive mask to ColoAttention, including cache offsets.

    Fully masked query rows (e.g. left padding) are unused by the caller's loss.
    Unmask those rows to satisfy the custom-mask kernel's nonempty-row contract;
    valid queries retain their original causal, padding and sliding constraints.
    """
    import torch

    from .layer import AttnMaskType

    if attention_mask is None:
        return {}
    if attention_mask.dtype == torch.bool:
        raise TypeError("ColoAttention requires an additive floating-point HF mask")
    empty_rows = (attention_mask == torch.finfo(attention_mask.dtype).min) | torch.isneginf(attention_mask)
    attention_mask = attention_mask.masked_fill(empty_rows.all(dim=-1, keepdim=True), 0)
    return {"attention_mask": attention_mask, "attention_mask_type": AttnMaskType.CUSTOM}


def get_obj_list_element(obj, attr: str):
    r"""
    Get the element of the list in the object

    If the attr is a normal attribute, return the attribute of the object.
    If the attr is a index type, return the element of the index in the list, like `layers[0]`.

    Args:
        obj (Object): The object to get
        attr (str): The suffix of the attribute to get

    """
    re_pattern = r"\[\d+\]"
    prog = re.compile(re_pattern)
    result = prog.search(attr)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace("[", "")
        matched_index = matched_index.replace("]", "")
        attr_ = attr.replace(matched_brackets, "")
        container_obj = getattr(obj, attr_)
        obj = container_obj[int(matched_index)]
    else:
        obj = getattr(obj, attr)
    return obj


def set_obj_list_element(obj, attr: str, value):
    r"""
    Set the element to value of a list object

    It used like set_obj_list_element(obj, 'layers[0]', new_layer), it will set obj.layers[0] to value

    Args:
        obj (object): The object to set
        attr (str): the string including a list index like `layers[0]`
    """
    re_pattern = r"\[\d+\]"
    prog = re.compile(re_pattern)
    result = prog.search(attr)
    if result:
        matched_brackets = result.group()
        matched_index = matched_brackets.replace("[", "")
        matched_index = matched_index.replace("]", "")
        attr_ = attr.replace(matched_brackets, "")
        container_obj = getattr(obj, attr_)
        container_obj[int(matched_index)] = value
    else:
        setattr(obj, attr, value)


def hasattr_(obj, attr: str):
    r"""
    Check whether the object has the multi sublevel attr

    Args:
        obj (object): The object to check
        attr (str): The multi level attr to check
    """
    attrs = attr.split(".")
    for a in attrs:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            return False
    return True


def setattr_(obj, attr: str, value, ignore: bool = False):
    r"""
    Set the object's multi sublevel attr to value, if ignore, ignore when it doesn't exist

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        value (Any): The value to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split(".")
    for a in attrs[:-1]:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return
            raise AttributeError(f"Object {obj.__class__.__name__} has no attribute {attr}")
    set_obj_list_element(obj, attrs[-1], value)


def getattr_(obj, attr: str, ignore: bool = False):
    r"""
    Get the object's multi sublevel attr

    Args:
        obj (object): The object to set
        attr (str): The multi level attr to set
        ignore (bool): Whether to ignore when the attr doesn't exist
    """

    attrs = attr.split(".")
    for a in attrs:
        try:
            obj = get_obj_list_element(obj, a)
        except AttributeError:
            if ignore:
                return None
            raise AttributeError(f"Object {obj.__class__.__name__} has no attribute {attr}")
    return obj
