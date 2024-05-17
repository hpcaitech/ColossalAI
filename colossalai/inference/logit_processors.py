# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/generation/logits_process.py
import logging
from typing import List, Union

import torch
import torch.nn.functional as F

_LOGITS_PROCESSOR_MAP = {}


def register_logits_processor(process_type):
    """
    register flops computation function for operation.
    """

    def register(func):
        global _LOGITS_PROCESSOR_MAP
        _LOGITS_PROCESSOR_MAP[process_type] = func
        return func

    return register


@register_logits_processor("no_repeat_ngram_size")
def apply_no_repeat_ngram_size(logits, ngram_size: int, batch_token_ids: List[List[int]]):
    """
    enforces no repetition of n-grams to avoid repetitions of word sequences.
    """

    if not isinstance(ngram_size, int) or ngram_size < 0:
        raise ValueError(f"'temperature={ngram_size}' should be a strictly positive integer.")

    if ngram_size != 0:
        batch_size = len(batch_token_ids)

        for batch_id in range(batch_size):
            current_token_ids = batch_token_ids[batch_id]
            current_len = len(current_token_ids)
            if current_len + 1 < ngram_size:
                continue

            ngrams_dict = {}

            for ngram in zip(*[current_token_ids[i:] for i in range(ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                ngrams_dict[prev_ngram_tuple] = ngrams_dict.get(prev_ngram_tuple, []) + [ngram[-1]]

            prev_ngrams = tuple(current_token_ids[current_len + 1 - ngram_size : current_len])
            banned_token = ngrams_dict.get(prev_ngrams, [])

            logits[batch_id, banned_token] = -float("inf")

    return logits


@register_logits_processor("repetition_penalty")
def apply_repetition_penalty(logits, penalty: float, batch_token_ids: List[List[int]]):
    """
    apply the penalty to the tokens present in the prompt.
    """

    if not isinstance(penalty, float) or not (penalty > 0):
        raise ValueError(f"'penalty={penalty}' has to be a strictly positive float and greater than 0.")

    logits_list = []

    # TODO(yuehuayingxueluo) This is only a temporary implementation. Later, we will implement presence_penalties, frequency_penalties, and repetition_penalties using CUDA kernels.
    if penalty != 1.0:
        for batch_id in range(len(batch_token_ids)):
            current_logit = logits[batch_id]
            current_token = torch.tensor(batch_token_ids[batch_id], dtype=torch.long, device=logits.device)

            curretn_socre = torch.gather(current_logit, 0, current_token)
            curretn_socre = torch.where(curretn_socre < 0, curretn_socre * penalty, curretn_socre / penalty)
            logits_list.append(current_logit.scatter(0, current_token, curretn_socre))

        logits = torch.stack(logits_list)

    return logits


@register_logits_processor("temperature")
def apply_temperature(logits, temperature: float):
    """
    apply temperature scaling.
    """

    if not isinstance(temperature, float) or not (0.0 < temperature <= 1.0):
        except_msg = f"'temperature={temperature}' should be a strictly positive float, less than or equal to 1.0 and greater than 0."
        if temperature == 0.0:
            except_msg += "if you want to use greedy decoding strategies, set `do_sample=False`."
        raise ValueError(except_msg)

    return logits if temperature == 1.0 else logits / temperature


@register_logits_processor("top_k")
def apply_top_k(logits, top_k: int):
    """
    top_k logit processor
    """

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"`top_k` should be a strictly positive integer, but got {top_k}.")

    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float("inf")
    return logits


@register_logits_processor("top_p")
def apply_top_p(logits, top_p: float):
    """
    top_p logit processor
    """

    if top_p < 0 or top_p > 1.0:
        raise ValueError(f"`top_p` should be a float > 0 and < 1, but got {top_p}.")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    sorted_indices_to_remove = torch.roll(sorted_indices_to_remove, 1, -1)
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = -float("inf")
    return logits


@register_logits_processor("forced_eos_token_id")
def apply_forced_eos_token_id(
    logits: torch.Tensor,
    sequence_lengths: Union[torch.Tensor, List[int]],
    max_lengths: Union[torch.Tensor, List[int]],
    eos_token_id: Union[int, List[int]],
):
    """
    Enforces the specified token as the last generated token when the maximum output length
    is reached. Notice that the maximum output lengths for different sequences, even if they're
    in the same batch, can be different.

    Args:
        logits(torch.Tensor): logits
        sequence_lengths(torch.Tensor): sequence lengths including prompt and output tokens
        max_lengths(torch.Tensor): the maximum length for each sequence
        eos_token_id(Union[int, List[int]]): forced eos token id
    """
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if isinstance(sequence_lengths, torch.Tensor):
        sequence_lengths = sequence_lengths.tolist()
    if isinstance(max_lengths, torch.Tensor):
        max_lengths = max_lengths.tolist()

    select_indexes = []
    num_sequences = logits.shape[0]
    sequence_lengths = sequence_lengths[:num_sequences]
    max_lengths = max_lengths[:num_sequences]
    for i, (sequence_length, max_out_length) in enumerate(zip(sequence_lengths, max_lengths)):
        if sequence_length == max_out_length - 1:
            select_indexes.append(i)
    if select_indexes:
        logits[select_indexes, :] = -float("inf")
        logits[select_indexes, eos_token_id] = 0

    return logits


def get_logits_processor(processor: str, logits, *args, **kwargs):
    """
    do logit process for given logits.

    Args:
        processor(str): the type of logit processor
        logits(torch.Tensor): input logits

    Returns:
        logits after process
    """
    if processor not in _LOGITS_PROCESSOR_MAP:
        logging.warning(f"Unsupported processor {processor}. Fall back to the original logits.")
    else:
        func = _LOGITS_PROCESSOR_MAP[processor]
        logits = func(logits, *args, **kwargs)

    return logits
