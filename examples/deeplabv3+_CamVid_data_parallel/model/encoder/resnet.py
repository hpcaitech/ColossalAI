""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
from copy import deepcopy

import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


new_settings = {
    "resnet18": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth"
    },
    "resnet50": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth"
    },
    "resnext50_32x4d": {
        "imagenet": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",
    },
    "resnext101_32x4d": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth"
    },
    "resnext101_32x8d": {
        "imagenet": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",
    },
    "resnext101_32x16d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",
    },
    "resnext101_32x32d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
    },
    "resnext101_32x48d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
    }
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext50_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x8d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x16d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x32d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x48d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
