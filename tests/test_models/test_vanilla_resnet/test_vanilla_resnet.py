#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pytest
import torch
import torchvision.models as models

from colossalai.builder import build_model

NUM_CLS = 10

RESNET18 = dict(
    type='VanillaResNet',
    block_type='ResNetBasicBlock',
    layers=[2, 2, 2, 2],
    num_cls=NUM_CLS
)

RESNET34 = dict(
    type='VanillaResNet',
    block_type='ResNetBasicBlock',
    layers=[3, 4, 6, 3],
    num_cls=NUM_CLS
)

RESNET50 = dict(
    type='VanillaResNet',
    block_type='ResNetBottleneck',
    layers=[3, 4, 6, 3],
    num_cls=NUM_CLS
)

RESNET101 = dict(
    type='VanillaResNet',
    block_type='ResNetBottleneck',
    layers=[3, 4, 23, 3],
    num_cls=NUM_CLS
)

RESNET152 = dict(
    type='VanillaResNet',
    block_type='ResNetBottleneck',
    layers=[3, 8, 36, 3],
    num_cls=NUM_CLS
)


def compare_model(data, colossal_model, torchvision_model):
    colossal_output = colossal_model(data)
    torchvision_output = torchvision_model(data)
    assert colossal_output[
               0].shape == torchvision_output.shape, f'{colossal_output[0].shape}, {torchvision_output.shape}'


@pytest.mark.cpu
def test_vanilla_resnet():
    """Compare colossal resnet with torchvision resnet"""
    # data
    x = torch.randn((2, 3, 224, 224))

    # resnet 18
    col_resnet18 = build_model(RESNET18)
    col_resnet18.build_from_cfg()
    torchvision_resnet18 = models.resnet18(num_classes=NUM_CLS)

    compare_model(x, col_resnet18, torchvision_resnet18)

    # resnet 34
    col_resnet34 = build_model(RESNET34)
    col_resnet34.build_from_cfg()
    torchvision_resnet34 = models.resnet34(num_classes=NUM_CLS)

    compare_model(x, col_resnet34, torchvision_resnet34)

    # resnet 50
    col_resnet50 = build_model(RESNET50)
    col_resnet50.build_from_cfg()
    torchvision_resnet50 = models.resnet50(num_classes=NUM_CLS)

    compare_model(x, col_resnet50, torchvision_resnet50)

    # resnet 101
    col_resnet101 = build_model(RESNET101)
    col_resnet101.build_from_cfg()
    torchvision_resnet101 = models.resnet101(num_classes=NUM_CLS)

    compare_model(x, col_resnet101, torchvision_resnet101)

    # # resnet 152
    col_resnet152 = build_model(RESNET152)
    col_resnet152.build_from_cfg()
    torchvision_resnet152 = models.resnet152(num_classes=NUM_CLS)

    compare_model(x, col_resnet152, torchvision_resnet152)


if __name__ == '__main__':
    test_vanilla_resnet()
