import json
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator


def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    reward_eps=5,
) -> torch.Tensor:
    """
    Args:
        log_probs: [batch_size, response_length]
        log_probs_base: [batch_size, response_length]
        action_mask: [batch_size, response_length]
        r: float
    Returns:
        reward: [batch_size, response_length]
    """
    log_ratio = log_probs - log_probs_base  # address numerical instability issue
    kl = -kl_coef * log_ratio * action_mask
    reward = kl
    r_clip = torch.clamp(r, -reward_eps, reward_eps)
    for i in range(action_mask.size(0)):
        assert action_mask[i].sum() > 0
        reward[i, : action_mask[i].sum()] += r_clip[i]
        reward[i, action_mask[i].sum() :] *= 0
    return reward, ((log_ratio * (log_ratio < 10)).exp() - 1 - log_ratio) * action_mask


def _log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def calc_action_log_probs(logits: torch.Tensor, sequences: torch.LongTensor, num_actions: int) -> torch.Tensor:
    """Calculate action log probs.

    Args:
        output (torch.Tensor): Output tensor of Actor.forward.logits.
        sequences (torch.LongTensor): Input sequences.
        num_actions (int): Number of actions.

    Returns:
        torch.Tensor: Action log probs.
    """
    log_probs = _log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
    return log_probs[:, -num_actions:]


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


def calc_masked_log_probs(logits: torch.Tensor, sequences: torch.LongTensor, mask: torch.Tensor) -> torch.Tensor:
    log_probs = _log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
    return log_probs * mask


def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)


def save_checkpoint(
    save_dir: Union[str, os.PathLike],
    booster: Booster,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    batch_size: int,
    coordinator: DistCoordinator,
) -> None:
    """
    Save model checkpoint, optimizer, LR scheduler and intermedidate running states.
    """

    save_dir = os.path.join(save_dir, f"epoch-{epoch}_step-{step}")
    os.makedirs(os.path.join(save_dir, "modeling"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "modeling"), shard=True)

    booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load_checkpoint(
    load_dir: Union[str, os.PathLike],
    booster: Booster,
    model: torch.nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
) -> Tuple[int, int, int]:
    """
    Load model checkpoint, optimizer, LR scheduler and intermedidate running states.
    """

    # Update booster params states.
    booster.load_model(model=model, checkpoint=os.path.join(load_dir, "modeling"))
    booster.load_optimizer(optimizer=optimizer, checkpoint=os.path.join(load_dir, "optimizer"))
    booster.load_lr_scheduler(lr_scheduler=lr_scheduler, checkpoint=os.path.join(load_dir, "lr_scheduler"))

    running_states = load_json(file_path=os.path.join(load_dir, "running_states.json"))
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
    )


def disable_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
