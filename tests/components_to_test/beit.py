import torch
from timm.models.beit import Beit

from colossalai.utils.cuda import get_current_device

from .registry import non_distributed_component_funcs
from .utils.dummy_data_generator import DummyDataGenerator


class DummyDataLoader(DummyDataGenerator):
    img_size = 64
    num_channel = 3
    num_class = 10
    batch_size = 4

    def generate(self):
        data = torch.randn((DummyDataLoader.batch_size, DummyDataLoader.num_channel, DummyDataLoader.img_size,
                            DummyDataLoader.img_size),
                           device=get_current_device())
        label = torch.randint(low=0,
                              high=DummyDataLoader.num_class,
                              size=(DummyDataLoader.batch_size,),
                              device=get_current_device())
        return data, label


@non_distributed_component_funcs.register(name='beit')
def get_training_components():

    def model_buider(checkpoint=False):
        model = Beit(img_size=DummyDataLoader.img_size,
                     num_classes=DummyDataLoader.num_class,
                     embed_dim=32,
                     depth=2,
                     num_heads=4)
        return model

    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()

    criterion = torch.nn.CrossEntropyLoss()
    return model_buider, trainloader, testloader, torch.optim.Adam, criterion
