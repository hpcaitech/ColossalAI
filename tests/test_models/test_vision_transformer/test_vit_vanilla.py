#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path

import pytest
import torch

from colossalai.builder import build_model
from colossalai.context import Config

CONFIG_PATH = Path(__file__).parent.joinpath('configs/vanilla_vit.py')


@pytest.mark.cpu
def test_with_vanilla_vit_config():
    config = Config.from_file(CONFIG_PATH)
    model = build_model(config.model)
    model.build_from_cfg()

    img = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    out = model(img)
    loss = out.mean()
    loss.backward()


if __name__ == '__main__':
    test_with_vanilla_vit_config()
