from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from colossalai.device.device_mesh import DeviceMesh

__all__ = ['Plugin']


class Plugin:

    @property
    def supported_devices(self) -> List[torch.device]:
        pass

    @property
    def supported_precisions(self) -> List[str]:
        pass

    @property
    def control_precision(self) -> bool:
        pass

    @property
    def control_device(self) -> bool:
        pass

    @property
    def support_no_sync(self) -> bool:
        pass

    def setup_model(self, model: nn.Module, device_mesh_pool: DeviceMesh) -> nn.Module:
        pass

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        pass

    def setup_dataloader(self, dataloader: DataLoader) -> DataLoader:
        pass

    @property
    def device_mesh_shape(self) -> List[Tuple[int, ...]]:
        pass
