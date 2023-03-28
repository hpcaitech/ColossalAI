from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from transformers.generation_logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except ImportError:
    from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper


def prepare_logits_processor(top_k: Optional[int] = None,
                             top_p: Optional[float] = None,
                             temperature: Optional[float] = None) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        processor_list.append(TopKLogitsWarper(top_k))
    if top_p is not None and top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    return processor_list


def _is_sequence_finished(unfinished_sequences: torch.Tensor) -> bool:
    if dist.is_initialized() and dist.get_world_size() > 1:
        # consider DP
        unfinished_sequences = unfinished_sequences.clone()
        dist.all_reduce(unfinished_sequences)
    return unfinished_sequences.max() == 0


def sample(model: nn.Module,
           input_ids: torch.Tensor,
           max_length: int,
           early_stopping: bool = False,
           eos_token_id: Optional[int] = None,
           pad_token_id: Optional[int] = None,
           top_k: Optional[int] = None,
           top_p: Optional[float] = None,
           temperature: Optional[float] = None,
           prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
           update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
           **model_kwargs) -> torch.Tensor:
    if input_ids.size(1) >= max_length:
        return input_ids

    logits_processor = prepare_logits_processor(top_k, top_p, temperature)
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

    for _ in range(input_ids.size(1), max_length):
        model_inputs = prepare_inputs_fn(input_ids, **model_kwargs) if prepare_inputs_fn is not None else {
            'input_ids': input_ids
        }
        outputs = model(**model_inputs)

        next_token_logits = outputs['logits'][:, -1, :]
        # pre-process distribution
        next_token_logits = logits_processor(input_ids, next_token_logits)
        # sample
        probs = torch.softmax(next_token_logits, dim=-1, dtype=torch.float)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if update_model_kwargs_fn is not None:
            model_kwargs = update_model_kwargs_fn(outputs, **model_kwargs)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished if early_stopping=True
        if early_stopping and _is_sequence_finished(unfinished_sequences):
            break

    return input_ids


def generate(model: nn.Module,
             input_ids: torch.Tensor,
             max_length: int,
             num_beams: int = 1,
             do_sample: bool = True,
             early_stopping: bool = False,
             eos_token_id: Optional[int] = None,
             pad_token_id: Optional[int] = None,
             top_k: Optional[int] = None,
             top_p: Optional[float] = None,
             temperature: Optional[float] = None,
             prepare_inputs_fn: Optional[Callable[[torch.Tensor, Any], dict]] = None,
             update_model_kwargs_fn: Optional[Callable[[dict, Any], dict]] = None,
             **model_kwargs) -> torch.Tensor:
    """Generate token sequence. The returned sequence is input_ids + generated_tokens.

    Args:
        model (nn.Module): model
        input_ids (torch.Tensor): input sequence
        max_length (int): max length of the returned sequence
        num_beams (int, optional): number of beams. Defaults to 1.
        do_sample (bool, optional): whether to do sample. Defaults to True.
        early_stopping (bool, optional): if True, the sequence length may be smaller than max_length due to finding eos. Defaults to False.
        eos_token_id (Optional[int], optional): end of sequence token id. Defaults to None.
        pad_token_id (Optional[int], optional): pad token id. Defaults to None.
        top_k (Optional[int], optional): the number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
        top_p (Optional[float], optional): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
        temperature (Optional[float], optional): The value used to module the next token probabilities. Defaults to None.
        prepare_inputs_fn (Optional[Callable[[torch.Tensor, Any], dict]], optional): Function to preprocess model inputs. Arguments of this function should be input_ids and model_kwargs. Defaults to None.
        update_model_kwargs_fn (Optional[Callable[[dict, Any], dict]], optional): Function to update model_kwargs based on outputs. Arguments of this function should be outputs and model_kwargs. Defaults to None.
    """
    is_greedy_gen_mode = ((num_beams == 1) and do_sample is False)
    is_sample_gen_mode = ((num_beams == 1) and do_sample is True)
    is_beam_gen_mode = ((num_beams > 1) and do_sample is False)
    if is_greedy_gen_mode:
        # run greedy search
        raise NotImplementedError
    elif is_sample_gen_mode:
        # run sample
        return sample(model,
                      input_ids,
                      max_length,
                      early_stopping=early_stopping,
                      eos_token_id=eos_token_id,
                      pad_token_id=pad_token_id,
                      top_k=top_k,
                      top_p=top_p,
                      temperature=temperature,
                      prepare_inputs_fn=prepare_inputs_fn,
                      update_model_kwargs_fn=update_model_kwargs_fn,
                      **model_kwargs)
    elif is_beam_gen_mode:
        raise NotImplementedError
    else:
        raise ValueError("Unsupported generation mode")
