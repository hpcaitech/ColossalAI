from typing import Any, Dict, List

import torch


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
