from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedTokenizer
from transformers.generation.utils import GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_prompt_template(
    input_query: str,
    history: List[Dict] = None,
    roles: list = ["", "Human", "Assistant"],
) -> str:
    """
    Generates a prompt template for chat models based on input and history.

    Args:
        input_query (str): User's current input query.
        history (List[Dict], optional): List of past conversations, each a dict with 'role' and 'message'.
        roles (list): Specifies the roles in the conversation, defaults to ["", "Human", "Assistant"].

    Returns:
        str: A formatted prompt including the input query and history.
    """
    prompt = ""
    if history is None:
        new_history = []
    else:
        new_history = deepcopy(history)

    new_history.append({"role": roles[1], "message": input_query.strip()})
    new_history.append({"role": roles[2], "message": None})

    for _, item in enumerate(new_history):
        role = item.get("role")
        message = item.get("message")
        if role == roles[0]:
            prompt += f"<s>{message}\n\n"
        else:
            if message:
                prompt += f"{role}: <s>{message}</s>"
            else:
                prompt += f"{role}: <s>"
    return prompt


@torch.inference_mode()
def streaming_chat(
    model: Any,
    tokenizer: PreTrainedTokenizer,
    input_query: str,
    history: List[Dict] = None,
    roles: list = ["", "Human", "Assistant"],
    past_key_values: Tuple[Tuple[torch.FloatTensor, Any], Any] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    length_penalty: float = 1.2,
    max_new_tokens: int = 512,
    logits_processor: LogitsProcessorList = None,
    return_past_key_values: bool = False,
    **kwargs,
):
    """
    Streaming chat responses generation with a given model and tokenizer.

    Args:
        model (Any): The language model to generate responses.
        tokenizer (PreTrainedTokenizer): Tokenizer compatible with the model, used for encoding inputs and decoding responses.
        input_query (str): The current user input to respond to.
        history (List[Dict], optional): A list of past conversations, where each conversation is a dictionary with keys 'role' and 'message'.
        roles (list): Roles involved in the conversation, defaults to ["", "Human", "Assistant"].
        past_key_values (Tuple[Tuple[torch.FloatTensor, Any], Any], optional): Past key values for incremental decoding.
        temperature (float): The temperature value for token sampling, defaults to 0.8.
        top_p (float): Nucleus sampling probability threshold, defaults to 0.95.
        top_k (int): Top-K filtering threshold, defaults to 50.
        do_sample (bool): Whether to sample responses, defaults to True.
        length_penalty (float): Penalty for response length, defaults to 1.2.
        max_new_tokens (int): Maximum number of new tokens to generate, defaults to 512.
        logits_processor (LogitsProcessorList, optional): Custom logits processors, defaults to None.
        return_past_key_values (bool): Whether to return past key values for further incremental decoding, defaults to False.
        **kwargs: Additional keyword arguments for generation.

    Yields:
        Tuple[str, List[Dict], Optional[Tuple[Tuple[torch.FloatTensor, Any], Any]]]: A tuple containing the generated response, updated history, and
        optionally the updated past key values if `return_past_key_values` is True.

    Ensures padding is on the left side for the tokenizer.
    """
    assert tokenizer.padding_side == "left", "Current generation only supports left padding."
    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()

    generation_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "length_penalty": length_penalty,
        "use_cache": True,
        **kwargs,
    }

    prompt_str = get_prompt_template(input_query, history=history, roles=roles)

    eos_token_id = [tokenizer.eos_token_id]
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    history.append({"role": roles[1], "message": input_query.strip()})
    history.append({"role": roles[2], "message": None})

    for outputs in stream_generate(
        model,
        **inputs,
        past_key_values=past_key_values,
        eos_token_id=eos_token_id,
        return_past_key_values=return_past_key_values,
        **generation_kwargs,
    ):
        if return_past_key_values:
            outputs, past_key_values = outputs

        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) : -1]
        response = tokenizer.decode(outputs)

        history[-1]["message"] = response.strip()
        if return_past_key_values:
            yield response, history, past_key_values
        else:
            yield response, history


@torch.inference_mode()
def stream_generate(
    model: Any,
    input_ids: torch.Tensor,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    return_past_key_values: bool = False,
    **kwargs,
):
    """
    Generates sequences of token ids using the specified model and generation parameters.
    Adapted from https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py

    Args:
        model (Any): The model used for generating sequences of token ids.
        input_ids (torch.Tensor): The sequence used as a prompt for the generation or as model inputs to the encoder.
        generation_config (Optional[GenerationConfig]): The generation configuration to be used as base parametrization for the generation call.
        logits_processor (Optional[LogitsProcessorList]): Custom logits processors that complement the default logits processors built from arguments
        and generation config.
        stopping_criteria (Optional[StoppingCriteriaList]): Custom stopping criteria that complement the default stopping criteria built from arguments
        and a generation config.
        prefix_allowed_tokens_fn (Optional[Callable[[int, torch.Tensor], List[int]]]): Function to constrain token generation.
        return_past_key_values (bool): Whether to return past key values for further incremental decoding, defaults to False.
        **kwargs: Additional parameters for model generation.

    Yields:
        torch.Tensor: The generated token IDs, updated after each generation step.
        Optional[Tuple[Tuple[torch.FloatTensor, Any], Any]]: The past key values, returned if `return_past_key_values` is True, defaults to False.
    """
    input_ids_len = input_ids.size(1)

    if generation_config is None:
        generation_config = model.generation_config
    generation_config = deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)

    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    if generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_len

    if input_ids_len >= generation_config.max_length:
        input_ids_string = "decoder_input_ids" if model.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_len}, but `max_length` is set to"
            f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
            " increasing `max_new_tokens`."
        )
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    # prepare distribution pre_processing samplers
    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_len,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    # prepare stopping criteria
    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    logits_warper = model._get_logits_warper(generation_config)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None

    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        # NOTE: this is correct only in left padding mode
        # pre-process distribution
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
        )

        if return_past_key_values:
            yield input_ids, outputs.past_key_values
        else:
            yield input_ids
        # stop when each sentence is finished, or if exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break
