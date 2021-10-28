# from colossal.components.optimizer.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmupLR, FlatAnnealingLR, FlatAnnealingWarmupLR
# from colossal.components.optimizer.lr_scheduler import LinearWarmupLR
# from colossal.components.optimizer.lr_scheduler import MultiStepLR, MultiStepWarmupLR
# from colossal.components.optimizer.lr_scheduler import OneCycleLR
# from colossal.components.optimizer.lr_scheduler import PolynomialLR, PolynomialWarmupLR
import matplotlib.pyplot as plt
import pytest
from torch.optim import SGD
from torchvision.models import resnet18

from colossalai.builder import build_lr_scheduler

NUM_EPOCHS = 5
NUM_STEPS_PER_EPOCH = 10

cfg = {
    'warmup_steps': 5
}


def init_cfg(name, **kwargs):
    return {
        'type': name,
        **cfg,
        **kwargs
    }


def test_scheduler(optimizer, scheduler_name, **kwargs):
    for group in optimizer.param_groups:
        group['lr'] = 0.1
    config = init_cfg(scheduler_name, **kwargs)
    scheduler = build_lr_scheduler(config,
                                   optimizer, NUM_EPOCHS * NUM_STEPS_PER_EPOCH, NUM_STEPS_PER_EPOCH)
    x = []
    y = []
    for epoch in range(NUM_EPOCHS):
        for i in range(NUM_STEPS_PER_EPOCH):
            step = epoch * NUM_STEPS_PER_EPOCH + i
            lr = optimizer.param_groups[0]['lr']
            x.append(step)
            y.append(lr)
            scheduler.step()
    print(y)
    plt.plot(x, y)
    plt.show()


@pytest.mark.skip("This test is skipped as it requires visualization, "
                  "You can visualize the test output plots on your local environment")
def test():
    model = resnet18()
    optimizer = SGD(model.parameters(), lr=1.0)
    test_scheduler(optimizer, 'CosineAnnealingLR')
    test_scheduler(optimizer, 'CosineAnnealingWarmupLR')
    test_scheduler(optimizer, 'FlatAnnealingLR')
    test_scheduler(optimizer, 'FlatAnnealingWarmupLR')
    test_scheduler(optimizer, 'LinearWarmupLR')
    test_scheduler(optimizer, 'MultiStepLR', milestones=[1, 3])
    test_scheduler(optimizer, 'MultiStepWarmupLR', milestones=[1, 3])
    test_scheduler(optimizer, 'MultiStepWarmupLR',
                   milestones=[1, 3], warmup_epochs=1)
    test_scheduler(optimizer, 'PolynomialLR', power=2.0)
    test_scheduler(optimizer, 'PolynomialWarmupLR', power=2.0)
    test_scheduler(optimizer, 'OneCycleLR')


if __name__ == '__main__':
    test()
