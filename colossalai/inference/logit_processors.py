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


def logit_processor(processor: str, logits, attrs):
    """
    do logit process for given logits.

    Args:
        processor(str): the type of logit processor
        logits(torch.Tensor): input logits
        attrs(dict): attrs of the logit processor

    Returns:
        logits after process
    """
    if processor not in _LOGIT_PROCESSOR_MAP:
        return logits
    else:
        func = _LOGIT_PROCESSOR_MAP[processor]
        try:
            logits = func(logits, attrs)
        except Exception:
            return logits
        return logits
