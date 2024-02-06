import torch.nn as nn
from torch.optim import Adam

from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR


def test_lr_scheduler_save_load():
    model = nn.Linear(10, 10)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=5, warmup_steps=2)
    new_scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=5, warmup_steps=2)
    for _ in range(5):
        scheduler.step()
        state_dict = scheduler.state_dict()
        new_scheduler.load_state_dict(state_dict)
        assert state_dict == new_scheduler.state_dict()


if __name__ == "__main__":
    test_lr_scheduler_save_load()
