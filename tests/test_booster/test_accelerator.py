import pytest
import torch.nn as nn
from torchvision.models import resnet18

from colossalai.booster.accelerator import Accelerator


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_accelerator(device):
    acceleartor = Accelerator(device)
    model = nn.Linear(8, 8)
    model = acceleartor.configure_model(model)
    assert next(model.parameters()).device.type == device
