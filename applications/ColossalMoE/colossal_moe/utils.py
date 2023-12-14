import json
import os
from typing import Any, Dict, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def load_model(ckpt_path: str, model, booster: Booster, optimizer=None):
    # pytorch ckpt
    if os.path.exists(os.path.join(ckpt_path, "model.safetensors.index.json")):
        ckpt_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    # saved ckpt
    elif os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin.index.json")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
    # download
    else:
        ckpt_path = snapshot_download(ckpt_path)
    booster.load_model(model, ckpt_path)
    if optimizer is not None:
        optimizer.sync_moe_master_param()
        optimizer.update_master_params(model)


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
    load_model(os.path.join(load_dir, "modeling"), model, booster, optimizer)
    booster.load_optimizer(optimizer=optimizer, checkpoint=os.path.join(load_dir, "optimizer"))
    booster.load_lr_scheduler(lr_scheduler=lr_scheduler, checkpoint=os.path.join(load_dir, "lr_scheduler"))

    running_states = load_json(file_path=os.path.join(load_dir, "running_states.json"))
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
    )
