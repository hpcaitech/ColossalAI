import os

import torch

from huggingface_hub import snapshot_download
from colossalai.booster import Booster

def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

@torch.no_grad()
def load_ckpt(ckpt_path: str, model, booster: Booster, optimizer = None):
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
