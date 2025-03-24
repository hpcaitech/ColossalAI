from typing import Any, Dict, List

import torch

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


def split_into_microbatches(data_dict, microbatch_size):
    """
    将包含多个张量的字典根据 microbatch_size 切分成多个微批次字典。
    :param data_dict: 包含多个张量的字典，input_ids 形状为 (batch_size, seq_len, hidden_dim)
    :param microbatch_size: 每个微批次的大小
    :return: 微批次字典列表
    """
    batch_size = next(iter(data_dict.values())).size(0)
    microbatch_dicts = []

    for start_idx in range(0, batch_size, microbatch_size):
        end_idx = min(start_idx + microbatch_size, batch_size)
        microbatch_dict = {}
        for key, tensor in data_dict.items():
            if tensor.size(0) == batch_size:
                microbatch_dict[key] = tensor[start_idx:end_idx]
            else:
                microbatch_dict[key] = tensor
        microbatch_dicts.append(microbatch_dict)

    return microbatch_dicts


def filter_microbatch_dicts(microbatch_dicts):
    """
    遍历 microbatch_dicts 列表，移除每个字典中键不在 ("input_ids", "attention_mask") 范围内的键值对
    :param microbatch_dicts: 包含多个字典的列表
    :return: 过滤后的 microbatch_dicts 列表
    """
    filtered_dicts = []
    allowed_keys = ("input_ids", "attention_mask")
    for microbatch_dict in microbatch_dicts:
        filtered_dict = {key: value for key, value in microbatch_dict.items() if key in allowed_keys}
        filtered_dicts.append(filtered_dict)
    return filtered_dicts


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


def calc_action_log_probs(
    logits: torch.Tensor,
    sequences: torch.LongTensor,
    num_actions: int,
    shard_config,
    vocab_size: int = None,
) -> torch.Tensor:
    """Calculate action log probs.

    Args:
        logits (torch.Tensor): Output tensor of Actor.forward.logits.
        sequences (torch.LongTensor): Input sequences.
        num_actions (int): Number of actions.
        shard_config
        vocab_size


    Returns:
        torch.Tensor: Action log probs.
    """
    # labels: torch.Tensor,  # [B, S] or [B, S, Vocab_size]
    # logits: torch.Tensor,  # [B, S, Vocab_size]
    log_probs = dist_log_prob(sequences, logits, shard_config, vocab_size, logits.dtype)
    log_probs = log_probs.squeeze(-1)
    return log_probs[:, -num_actions:]


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
