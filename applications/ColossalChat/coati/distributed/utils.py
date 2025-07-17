import json
import os
from typing import Any, Dict, List

import torch
from filelock import FileLock

from colossalai.shardformer.layer.loss import dist_log_prob


def unbind_batch(batch: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    batches = []
    for k, v in batch.items():
        if len(batches) == 0:
            unbinded_tensors = v.unbind(0)
            batches = [{k: tensor} for tensor in unbinded_tensors]
        else:
            unbinded_tensors = v.unbind(0)
            assert len(batches) == len(unbinded_tensors)
            for i, tensor in enumerate(unbinded_tensors):
                batches[i][k] = tensor
    return batches


def bind_batch(batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = {}
    for k in batches[0].keys():
        batch[k] = torch.stack([batch[k] for batch in batches], dim=0)
    return batch


def pre_send(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # compress mask to save bandwidth
    if "attention_mask" in batch:
        batch["attention_mask"] = batch["attention_mask"].to(torch.bool)
    if "action_mask" in batch:
        batch["action_mask"] = batch["action_mask"].to(torch.bool)
    return batch


def post_recv(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # decompress mask
    if "attention_mask" in batch:
        batch["attention_mask"] = batch["attention_mask"].to(torch.int)
    if "action_mask" in batch:
        batch["action_mask"] = batch["action_mask"].to(torch.int)
    return batch


def update_by_default(data: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    data = data.copy()
    for k, v in default.items():
        if k not in data:
            data[k] = v
    return data


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the log probabilities from logits for the given labels.

    Args:
        logits (torch.Tensor): The input logits.
        labels (torch.Tensor): The target labels.

    Returns:
        torch.Tensor: The log probabilities corresponding to the labels.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    per_label_logps = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return per_label_logps.squeeze(-1)


def memory_efficient_logprob(
    logits: torch.Tensor,
    inputs: torch.Tensor,
    num_action: int,
    chunk_size: int = 2048,
    shard_config: Any = None,
    vocab_size: int = None,
) -> torch.Tensor:
    """
    Calculate action log probs in a memory-efficient way by processing in chunks.
    Args:
        logits (torch.Tensor): Output tensor of Actor.forward.logits.
        inputs (torch.LongTensor): Input sequences.
        num_action (int): Number of actions.
        chunk_size (int, optional): Size of each chunk to process. Default is 2048.
        shard_config: Shard configuration for distributed computation.
        vocab_size (int, optional): Vocabulary size. Default is None.
    Returns:
        torch.Tensor: Action log probs.
    """
    action_log_probs = torch.zeros((logits.size(0), num_action), device=logits.device, dtype=logits.dtype)
    context_length = logits.size(1) - num_action
    for i in range(action_log_probs.size(0)):
        # loop over each sample in the micro-batch
        for start in range(context_length, logits.size(1), chunk_size):
            end = min(start + chunk_size, logits.size(1))
            # calculate log probs in chunks to save memory
            log_probs = dist_log_prob(
                inputs[i : i + 1, start - 1 : end],
                logits[i : i + 1, start - 1 : end],
                shard_config,
                vocab_size,
                logits.dtype,
            )  # [1, chunk_size, 1]
            log_probs = log_probs.squeeze(-1)
            action_log_probs[i, start - context_length : end - context_length] += log_probs[0]
    return action_log_probs


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy
    Reference: https://github.com/volcengine/verl/blob/96b730bbed80292a439f0c0057d3920ab8b28d52/verl/utils/torch_functional.py#L145
    """
    p = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(p * logits, dim=-1)
    return entropy


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute the masked mean of a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        mask (torch.Tensor): The mask tensor with the same shape as the input tensor.
        dim (int, optional): The dimension along which to compute the mean. Default is 1.

    Returns:
        torch.Tensor: The masked mean tensor.

    """
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def masked_sum(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute the masked sum of a tensor along a specified dimension.

    Args:
        tensor (torch.Tensor): The input tensor.
        mask (torch.Tensor): The mask tensor with the same shape as the input tensor.
        dim (int, optional): The dimension along which to compute the sum. Default is 1.

    Returns:
        torch.Tensor: The masked sum tensor.

    """
    tensor = tensor * mask
    return tensor.sum(dim=dim)


def safe_append_to_jsonl_file(file_path, data):
    with FileLock(file_path + ".lock"):
        # Ensure file exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf8") as f:
            for entry in data:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + "\n")
