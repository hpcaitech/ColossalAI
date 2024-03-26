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


@register_logit_processor("top_k")
def top_k_logit_processor(logits, top_k: int):
    """
    top_k logit processor
    """
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float("inf")
    return logits


@register_logit_processor("top_p")
def top_p_logit_processor(logits, top_p: float):
    """
    top_p logit processor
    """
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
