from typing import Optional

import torch


def gpt_prepare_inputs_fn(input_ids: torch.Tensor, past: Optional[torch.Tensor] = None, **kwargs) -> dict:
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def update_model_kwargs_fn(outputs: dict, **model_kwargs) -> dict:
    if "past_key_values" in outputs:
        model_kwargs["past"] = outputs["past_key_values"]
    else:
        model_kwargs["past"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

    return model_kwargs


def opt_prepare_inputs_fn(input_ids: torch.Tensor,
                          past: Optional[torch.Tensor] = None,
                          attention_mask: Optional[torch.Tensor] = None,
                          use_cache: Optional[bool] = None,
                          **kwargs) -> dict:
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past:
        input_ids = input_ids[:, -1:]
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,    # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }


def bloom_prepare_inputs_fn(input_ids: torch.Tensor,
                            past: Optional[torch.Tensor] = None,
                            attention_mask: Optional[torch.Tensor] = None,
                            use_cache: Optional[bool] = None,
                            **kwargs) -> dict:
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past:
        input_ids = input_ids[:, -1:]
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,    # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past,
        "use_cache": use_cache,
    }
