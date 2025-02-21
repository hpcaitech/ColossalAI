from typing import Dict, List

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
    # compress attention_mask to save bandwidth
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        batch["attention_mask"] = attention_mask.to(torch.bool)
    return batch


def post_recv(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # decompress attention_mask
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        batch["attention_mask"] = attention_mask.to(torch.int)
    return batch
