# This code is adapted from huggingface transformers: https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/generation/logits_process.py
from typing import List

import torch
import torch.nn.functional as F

_LOGIT_PROCESSOR_MAP = {}


def register_logit_processor(process_type):
    """
    register flops computation function for operation.
    """

    def register(func):
        global _LOGIT_PROCESSOR_MAP
        _LOGIT_PROCESSOR_MAP[process_type] = func
        return func

    return register


@register_logit_processor("no_repeat_ngram_size")
def no_repeat_ngram_size_logit_process(logits, ngram_size: int, batch_token_ids: List[List[int]]):
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


@register_logit_processor("repetition_penalty")
def repetition_penalty_logit_process(logits, penalty: float, batch_token_ids: List[List[int]]):
    """
    apply the penalty to the tokens present in the prompt.
    """

    if not isinstance(penalty, float) or not (penalty > 0):
        raise ValueError(f"'penalty={penalty}' has to be a strictly positive float and greater than 0.")

    logit_list = []

    # TODO(yuehuayingxueluo) This is only a temporary implementation. Later, we will implement presence_penalties, frequency_penalties, and repetition_penalties using CUDA kernels.
    if penalty != 1.0:
        for batch_id in range(len(batch_token_ids)):
            current_logit = logits[batch_id]
            current_token = torch.tensor(batch_token_ids[batch_id], dtype=torch.long, device=logits.device)

            curretn_socre = torch.gather(current_logit, 0, current_token)
            curretn_socre = torch.where(curretn_socre < 0, curretn_socre * penalty, curretn_socre / penalty)
            logit_list.append(current_logit.scatter(0, current_token, curretn_socre))

        logits = torch.stack(logit_list)

    return logits


@register_logit_processor("temperature")
def temperature_logit_process(logits, temperature: float):
    """
    apply temperature scaling.
    """

    if not isinstance(temperature, float) or not (0.0 < temperature <= 1.0):
        except_msg = f"'temperature={temperature}' should be a strictly positive float, less than or equal to 1.0 and greater than 0."
        if temperature == 0.0:
            except_msg += "if you want to use greedy decoding strategies, set `do_sample=False`."
        raise ValueError(except_msg)

    return logits if temperature == 1.0 else logits / temperature


@register_logit_processor("top_k")
def top_k_logit_processor(logits, top_k: int):
    """
    top_k logit processor
    """

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"`top_k` should be a strictly positive integer, but got {top_k}.")

    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float("inf")
    return logits


@register_logit_processor("top_p")
def top_p_logit_processor(logits, top_p: float):
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


def logit_processor(processor: str, logits, *args, **kwargs):
    """
    do logit process for given logits.

    Args:
        processor(str): the type of logit processor
        logits(torch.Tensor): input logits

    Returns:
        logits after process
    """
    if processor not in _LOGIT_PROCESSOR_MAP:
        return logits
    else:
        func = _LOGIT_PROCESSOR_MAP[processor]
        logits = func(logits, *args, **kwargs)
        return logits
