from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

from colossalai.utils.cuda import get_current_device


class DummyDataGenerator(ABC):

    def __init__(self, length=10):
        self.length = length

    @abstractmethod
    def generate(self):
        pass

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


class DummyDataLoader(DummyDataGenerator):
    batch_size = 4
    channel = 3
    category = 8
    image_size = 224

    def generate(self):
        image_dict = {}
        image_dict['pixel_values'] = torch.rand(DummyDataLoader.batch_size,
                                                DummyDataLoader.channel,
                                                DummyDataLoader.image_size,
                                                DummyDataLoader.image_size,
                                                device=get_current_device()) * 2 - 1
        image_dict['label'] = torch.randint(DummyDataLoader.category, (DummyDataLoader.batch_size,),
                                            dtype=torch.int64,
                                            device=get_current_device())
        return image_dict


class ViTCVModel(nn.Module):

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 image_size=224,
                 patch_size=16,
                 num_channels=3,
                 num_labels=8,
                 checkpoint=False):
        super().__init__()
        self.checkpoint = checkpoint
        self.model = ViTForImageClassification(
            ViTConfig(hidden_size=hidden_size,
                      num_hidden_layers=num_hidden_layers,
                      num_attention_heads=num_attention_heads,
                      image_size=image_size,
                      patch_size=patch_size,
                      num_channels=num_channels,
                      num_labels=num_labels))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values)


def vit_base_s(checkpoint=True):
    return ViTCVModel(checkpoint=checkpoint)


def vit_base_micro(checkpoint=True):
    return ViTCVModel(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, checkpoint=checkpoint)


def get_training_components():
    trainloader = DummyDataLoader()
    testloader = DummyDataLoader()
    return vit_base_micro, trainloader, testloader, torch.optim.Adam, torch.nn.functional.cross_entropy
