import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointModule(nn.Module):
    def __init__(self, checkpoint: bool = False):
        super().__init__()
        self.checkpoint = checkpoint
        self._use_checkpoint = checkpoint

    def _forward(self, *args, **kwargs):
        raise NotImplementedError("CheckpointModule should implement _forward method instead of origin forward")

    def forward(self, *args, **kwargs):
        if self._use_checkpoint:
            return checkpoint(self._forward, *args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self._use_checkpoint = self.checkpoint
        return super().train(mode=mode)

    def eval(self):
        self._use_checkpoint = False
        return super().eval()
