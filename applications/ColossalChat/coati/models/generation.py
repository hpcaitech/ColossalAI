import copy
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from .utils import repad_to_left

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except ImportError:
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


def _prepare_logits_processor(
    top_k: Optional[int] = None, top_p: Optional[float] = None, temperature: Optional[float] = None
) -> LogitsProcessorList:
    """
    Prepare the logits processor list based on the given parameters.

    Args:
        top_k (Optional[int]): The number of highest probability logits to keep for each token.
        top_p (Optional[float]): The cumulative probability threshold for selecting tokens.
        temperature (Optional[float]): The temperature value to apply to the logits.

    Returns:
        LogitsProcessorList: The list of logits processors.

    """
    processor_list = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list


def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    """
    Check if the sequence generation is finished.

    Args:
        unfinished_sequences (torch.Tensor): Tensor indicating the unfinished sequences.

    Returns:
        bool: True if all sequences are finished, False otherwise.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        # consider DP
        unfinished_sequences = unfinished_sequences.clone()
        dist.all_reduce(unfinished_sequences)
    return unfinished_sequences.max() == 0


def update_model_kwargs_fn(outputs: dict, new_mask, **model_kwargs) -> dict:
    """
    Update the model keyword arguments based on the outputs and new mask.

    Args:
        outputs (dict): The outputs from the model.
        new_mask: The new attention mask.
        **model_kwargs: Additional model keyword arguments.

    Returns:
        dict: The updated model keyword arguments.
    """

    if "past_key_values" in outputs:
        model_kwargs["past_key_values"] = outputs["past_key_values"]
    else:
        model_kwargs["past_key_values"] = None

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    # update attention mask
    if "attention_mask" in model_kwargs:
        attention_mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat([attention_mask, new_mask], dim=-1)

    return model_kwargs


def prepare_inputs_fn(input_ids: torch.Tensor, **model_kwargs) -> dict:
    model_kwargs["input_ids"] = input_ids
    return model_kwargs


def _sample(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_length: int,
    early_stopping: bool = True,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    stop_token_ids: Optional[List[int]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_new_tokens: int = None,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    stream_interval: int = 2,
    **model_kwargs,
) -> torch.Tensor:
    """
    Generates new tokens using the given model and input_ids.

    Args:
        model (Any): The model used for token generation.
        input_ids (torch.Tensor): The input tensor containing the initial tokens.
        max_length (int): The maximum length of the generated tokens.
        early_stopping (bool, optional): Whether to stop generating tokens early if all sequences are finished. Defaults to True.
        eos_token_id (int, optional): The ID of the end-of-sequence token. Defaults to None.
        pad_token_id (int, optional): The ID of the padding token. Defaults to None.
        stop_token_ids (List[int], optional): A list of token IDs that, if encountered, will stop the generation process. Defaults to None.
        top_k (int, optional): The number of top-k tokens to consider during sampling. Defaults to None.
        top_p (float, optional): The cumulative probability threshold for top-p sampling. Defaults to None.
        temperature (float, optional): The temperature value for token sampling. Defaults to None.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to None.
        prepare_inputs_fn (Callable[[torch.Tensor, Any], dict], optional): A function to prepare the model inputs. Defaults to None.
        update_model_kwargs_fn (Callable[[dict, Any], dict], optional): A function to update the model kwargs. Defaults to None.
        stream_interval (int, optional): The interval for streaming generation. Defaults to 2.
        **model_kwargs: Additional keyword arguments for the model.

    Returns:
        torch.Tensor: The tensor containing the generated tokens.
    """
    context_length = input_ids.size(1)
    if max_new_tokens is None:
        max_new_tokens = max_length - context_length
    if context_length + max_new_tokens > max_length or max_new_tokens == 0:
        print("Exeeded length limitation")
        return input_ids
    logits_processor = _prepare_logits_processor(top_k, top_p, temperature)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    past = None
    for i in range(context_length, context_length + max_new_tokens):
        # Calculate attention mask
        if "attention_mask" not in model_kwargs:
            model_kwargs["attention_mask"] = input_ids.ne(pad_token_id)
        model_inputs = (
            prepare_inputs_fn(input_ids, past=past, **model_kwargs)
            if prepare_inputs_fn is not None
            else {"input_ids": input_ids, "attention_mask": input_ids.ne(pad_token_id)}
        )
        outputs = model(**model_inputs)

        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        # NOTE: this is correct only in left padding mode
        next_token_logits = outputs["logits"][:, -1, :]
        next_token_logits = logits_processor(input_ids, next_token_logits)

        # Sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        # if dist.get_rank() == 0:
        #     print(next_tokens[:1], tokenizer.decode(next_tokens[:1], skip_special_tokens=False), end=' ')

        # Finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # Update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, model_kwargs)

        # If eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        if stop_token_ids is not None:
            # If the last len(stop_token_ids) tokens of input_ids are equal to stop_token_ids, set sentence to finished.
            for stop_token_id in stop_token_ids:
                tokens_to_check = input_ids[:, -len(stop_token_id) :]
                unfinished_sequences = unfinished_sequences.mul(
                    torch.any(tokens_to_check != torch.LongTensor(stop_token_id).to(input_ids.device), dim=1).long()
                )

        # Stop when each sentence is finished if early_stopping=True
        if (early_stopping and _is_sequence_finished(unfinished_sequences)) or i == context_length + max_new_tokens - 1:
            # if i == context_length + max_new_tokens - 1:
            #     # Force to end with stop token ids
            #     stop_token_id = stop_token_ids[0]
            #     input_ids[input_ids[:, -1] != pad_token_id, -len(stop_token_id) :] = (
            #         torch.LongTensor(stop_token_id).to(input_ids.device).long()
            #     )
            return input_ids


@torch.inference_mode()
def generate(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_beams: int = 1,
    do_sample: bool = True,
    early_stopping: bool = True,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    **model_kwargs,
) -> torch.Tensor:
    """Generate token sequence. The returned sequence is input_ids + generated_tokens.

    Args:
        model (nn.Module): model
        input_ids (torch.Tensor): input sequence
        max_length (int): max length of the returned sequence
        num_beams (int, optional): number of beams. Defaults to 1.
        do_sample (bool, optional): whether to do sample. Defaults to True.
        early_stopping (bool, optional): if True, the sequence length may be smaller than max_length due to finding eos. Defaults to False.
        top_k (Optional[int], optional): the number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (Optional[float], optional): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
        temperature (Optional[float], optional): The value used to module the next token probabilities. Defaults to None.
        prepare_inputs_fn (Optional[Callable[[torch.Tensor, Any], dict]], optional): Function to preprocess model inputs. Arguments of this function should be input_ids and model_kwargs. Defaults to None.
        update_model_kwargs_fn (Optional[Callable[[dict, Any], dict]], optional): Function to update model_kwargs based on outputs. Arguments of this function should be outputs and model_kwargs. Defaults to None.
    """
    assert tokenizer.padding_side == "left", "Current generation only supports left padding."
    is_greedy_gen_mode = (num_beams == 1) and do_sample is False
    is_sample_gen_mode = (num_beams == 1) and do_sample is True
    is_beam_gen_mode = (num_beams > 1) and do_sample is False
    if is_greedy_gen_mode:
        raise NotImplementedError
    elif is_sample_gen_mode:
        # Run sample
        generation_kwargs = copy.deepcopy(model_kwargs)
        res = _sample(
            model,
            tokenizer,
            input_ids,
            max_length,
            early_stopping=early_stopping,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prepare_inputs_fn=prepare_inputs_fn,
            update_model_kwargs_fn=update_model_kwargs_fn,
            **generation_kwargs,
        )
        del generation_kwargs
        return res
    elif is_beam_gen_mode:
        raise NotImplementedError
    else:
        raise ValueError("Unsupported generation mode")


def _sample_streaming(
    model: Any,
    input_ids: torch.Tensor,
    max_length: int,
    early_stopping: bool = False,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    stop_token_ids: Optional[List[int]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    max_new_tokens: int = None,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    stream_interval: int = 2,
    **model_kwargs,
) -> torch.Tensor:
    """
    Generates new tokens using a streaming approach.

    Args:
        model (Any): The model used for token generation.
        input_ids (torch.Tensor): The input tensor containing the initial tokens.
        max_length (int): The maximum length of the generated sequence.
        early_stopping (bool, optional): Whether to stop generating tokens for a sequence if it is finished. Defaults to False.
        eos_token_id (int, optional): The ID of the end-of-sequence token. Defaults to None.
        pad_token_id (int, optional): The ID of the padding token. Defaults to None.
        stop_token_ids (List[int], optional): A list of token IDs that, if encountered, will mark the sequence as finished. Defaults to None.
        top_k (int, optional): The number of top-k tokens to consider during sampling. Defaults to None.
        top_p (float, optional): The cumulative probability threshold for top-p sampling. Defaults to None.
        temperature (float, optional): The temperature value for sampling. Defaults to None.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to None.
        prepare_inputs_fn (Callable[[torch.Tensor, Any], dict], optional): A function to prepare the model inputs. Defaults to None.
        update_model_kwargs_fn (Callable[[dict, Any], dict], optional): A function to update the model keyword arguments. Defaults to None.
        stream_interval (int, optional): The interval at which to yield the generated tokens. Defaults to 2.
        **model_kwargs: Additional keyword arguments to be passed to the model.

    Yields:
        torch.Tensor: The generated tokens at each step.

    Returns:
        torch.Tensor: The final generated tokens.
    """

    context_length = input_ids.size(1)
    if max_new_tokens is None:
        max_new_tokens = max_length - context_length
    if context_length + max_new_tokens > max_length or max_new_tokens == 0:
        return input_ids

    logits_processor = _prepare_logits_processor(top_k, top_p, temperature)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    past = None
    for i in range(context_length, context_length + max_new_tokens):
        # calculate attention mask
        if "attention_mask" not in model_kwargs:
            model_kwargs["attention_mask"] = input_ids.ne(pad_token_id)
        model_inputs = (
            prepare_inputs_fn(input_ids, past=past, **model_kwargs)
            if prepare_inputs_fn is not None
            else {"input_ids": input_ids, "attention_mask": input_ids.ne(pad_token_id)}
        )
        outputs = model(**model_inputs)
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        # NOTE: this is correct only in left padding mode
        next_token_logits = outputs["logits"][:, -1, :]
        next_token_logits = logits_processor(input_ids, next_token_logits)
        # sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        if stop_token_ids is not None:
            tokens_to_check = input_ids[:, -len(stop_token_ids) :]
            if isinstance(stop_token_ids[0], int):
                # If the last len(stop_token_ids) tokens of input_ids are equal to stop_token_ids, set sentence to finished.
                unfinished_sequences = unfinished_sequences.mul(
                    torch.any(tokens_to_check != torch.LongTensor(stop_token_ids).to(input_ids.device), dim=1).long()
                )
            else:
                for stop_token_id in stop_token_ids:
                    unfinished_sequences = unfinished_sequences.mul(
                        torch.any(tokens_to_check != torch.LongTensor(stop_token_id).to(input_ids.device), dim=1).long()
                    )

        # Stop when each sentence is finished if early_stopping=True
        if (
            (early_stopping and _is_sequence_finished(unfinished_sequences))
            or (i - context_length) % stream_interval == 0
            or i == context_length + max_new_tokens - 1
        ):
            yield input_ids
            if early_stopping and _is_sequence_finished(unfinished_sequences):
                break


@torch.inference_mode()
def generate_streaming(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_beams: int = 1,
    do_sample: bool = True,
    early_stopping: bool = False,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    **model_kwargs,
):
    """Generate token sequence. The returned sequence is input_ids + generated_tokens.

    Args:
        model (nn.Module): model
        input_ids (torch.Tensor): input sequence
        max_length (int): max length of the returned sequence
        num_beams (int, optional): number of beams. Defaults to 1.
        do_sample (bool, optional): whether to do sample. Defaults to True.
        early_stopping (bool, optional): if True, the sequence length may be smaller than max_length due to finding eos. Defaults to False.
        top_k (Optional[int], optional): the number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (Optional[float], optional): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
        temperature (Optional[float], optional): The value used to module the next token probabilities. Defaults to None.
        prepare_inputs_fn (Optional[Callable[[torch.Tensor, Any], dict]], optional): Function to preprocess model inputs. Arguments of this function should be input_ids and model_kwargs. Defaults to None.
        update_model_kwargs_fn (Optional[Callable[[dict, Any], dict]], optional): Function to update model_kwargs based on outputs. Arguments of this function should be outputs and model_kwargs. Defaults to None.
    """
    assert tokenizer.padding_side == "left", "Current generation only supports left padding."
    is_greedy_gen_mode = (num_beams == 1) and do_sample is False
    is_sample_gen_mode = (num_beams == 1) and do_sample is True
    is_beam_gen_mode = (num_beams > 1) and do_sample is False
    if is_greedy_gen_mode:
        # run greedy search
        raise NotImplementedError
    elif is_sample_gen_mode:
        # run sample
        for res in _sample_streaming(
            model,
            input_ids,
            max_length,
            early_stopping=early_stopping,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prepare_inputs_fn=prepare_inputs_fn,
            update_model_kwargs_fn=update_model_kwargs_fn,
            **model_kwargs,
        ):
            yield res
    elif is_beam_gen_mode:
        raise NotImplementedError
    else:
        raise ValueError("Unsupported generation mode")


def vllm_style_generate(*args, **kwargs):
    # passing stop token ids may not stop the generation properly due to BPE
    # Hack: decode first
    stops = kwargs.pop("stops")
    bs = args[1].size(0)
    stopped = []
    offset = args[1].size(1)
    tokenizer = args[2]
    decoded_response = ["" for _ in range(bs)]
    for output in generate_streaming(*args, **kwargs):
        res = tokenizer.batch_decode(output[:, offset:], skip_special_tokens=False)
        offset = output.size(1)
        for i in range(bs):
            if i in stopped:
                continue
            decoded_response[i] = decoded_response[i] + res[i]
            stop_detected = [stop for stop in stops if stop in decoded_response[i][-30:]]
            if len(stop_detected) > 0:
                stopped.append(i)
                decoded_response[i] = sorted(
                    [decoded_response[i].split(stop)[0] + stop for stop in stop_detected], key=lambda x: len(x)
                )[
                    0
                ]  # stop at the first met stopping condition
        if len(stopped) == bs:
            break
    return decoded_response


@torch.inference_mode()
def generate_tts(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_beams: int = 1,
    do_sample: bool = True,
    early_stopping: bool = True,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
    update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
    think_prefix: torch.Tensor = None,
    final_answer_prefix: str = None,
    thought_stop: str = None,
    final_answer_stop: str = None,
    num_ignore: int = 1,
    ignore_prefix: str = None,
    max_tokens_thinking: int = None,
    max_new_token_think: int = 1500,
    max_new_token_final_answer: int = 100,
    **model_kwargs,
) -> torch.Tensor:
    non_pad_indices_input_ids = []
    assert max_tokens_thinking < max_length
    for i in range(input_ids.size(0)):
        non_pad_indices_input_ids.append((input_ids[i] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0].min())

    o = torch.cat([input_ids, think_prefix.unsqueeze(dim=0).repeat_interleave(input_ids.size(0), dim=0)], dim=-1)
    prompt_len = o.size(1)
    model_kwargs["stops"] = [thought_stop]
    res = vllm_style_generate(
        model,
        o,
        tokenizer,
        max_tokens_thinking,
        num_beams,
        do_sample,
        early_stopping,
        top_k,
        top_p,
        temperature,
        prepare_inputs_fn,
        update_model_kwargs_fn,
        max_new_tokens=max_new_token_think,
        **model_kwargs,
    )
    if num_ignore > 0:
        res = [(r[: -len(thought_stop)] if r.endswith(thought_stop) else r) + ignore_prefix for r in res]
    else:
        res = [r + final_answer_prefix for r in res]
    tokenizer.padding_side = "right"
    new_tokens = tokenizer(res, padding=True, add_special_tokens=False, return_tensors="pt")["input_ids"].to(o.device)
    tokenizer.padding_side = "left"
    o = torch.cat([o[:, :prompt_len], new_tokens], dim=-1)
    # if dist.get_rank() == 0:
    #     decoded = tokenizer.decode(o[0],skip_special_tokens=False)
    #     print("###########\nfirst thought:\n",decoded)
    # Num of times to skip stop token
    for i in range(num_ignore):
        # add ignore prefix
        o = repad_to_left(o, tokenizer)  # repad for generation
        prompt_len = o.size(1)
        res = vllm_style_generate(
            model,
            o,
            tokenizer,
            max_length,
            num_beams,
            do_sample,
            early_stopping,
            top_k,
            top_p,
            temperature,
            prepare_inputs_fn,
            update_model_kwargs_fn,
            max_new_tokens=max_new_token_think,
            **model_kwargs,
        )
        if i != num_ignore - 1:
            res = [(r[: -len(thought_stop)] if r.endswith(thought_stop) else r) + ignore_prefix for r in res]
        else:
            res = [r + final_answer_prefix for r in res]
        tokenizer.padding_side = "right"
        new_tokens = tokenizer(res, padding=True, add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            o.device
        )
        tokenizer.padding_side = "left"
        o = torch.cat([o[:, :prompt_len], new_tokens], dim=-1)
        # if dist.get_rank() == 0:
        #     decoded = tokenizer.decode(o[0],skip_special_tokens=False)
        #     print(f"###########\n{i} thought:\n",decoded)
    o = repad_to_left(o, tokenizer)  # repad for generation
    prompt_len = o.size(1)
    model_kwargs["stops"] = [final_answer_stop]
    res = vllm_style_generate(
        model,
        o,
        tokenizer,
        max_length,
        num_beams,
        do_sample,
        early_stopping,
        top_k,
        top_p,
        temperature,
        prepare_inputs_fn,
        update_model_kwargs_fn,
        max_new_tokens=max_new_token_final_answer,
        **model_kwargs,
    )
    tokenizer.padding_side = "right"
    new_tokens = tokenizer(res, padding=True, add_special_tokens=False, return_tensors="pt")["input_ids"].to(o.device)
    tokenizer.padding_side = "left"
    o = torch.cat([o[:, :prompt_len], new_tokens], dim=-1)
    # if dist.get_rank() == 0:
    #     decoded = tokenizer.decode(o[0],skip_special_tokens=True)
    #     print(f"###########\nFinal Answer:\n",decoded)

    # repad to aligh with the left padding of input id
    max_left_padded_seq_len = 0
    padding_left = []
    starts_o = []
    ends_o = []
    for i in range(o.size(0)):
        non_pad_indices_o = (o[i] != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        start_o, end_o = non_pad_indices_o.min(), non_pad_indices_o.max()
        start_input_ids = non_pad_indices_input_ids[i]
        padding_left.append(start_input_ids)
        starts_o.append(start_o)
        ends_o.append(end_o)
        max_left_padded_seq_len = max(max_left_padded_seq_len, start_input_ids - start_o + end_o + 1)
    repaded_output = []
    for i, s, e, p in zip(range(o.size(0)), starts_o, ends_o, padding_left):
        repaded_output.append(
            F.pad(o[i][s : e + 1], (p, max_left_padded_seq_len - (p + e - s + 1)), value=tokenizer.pad_token_id)
        )
    repaded_output = torch.stack(repaded_output)
    return repaded_output
