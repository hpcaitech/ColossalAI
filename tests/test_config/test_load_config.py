#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from pathlib import Path

import pytest

from colossalai.context.config import Config
from colossalai.builder import build_ophooks


@pytest.mark.cpu
def test_load_config():
    filename = Path(__file__).parent.joinpath('sample_config.py')
    config = Config.from_file(filename)

    assert config.train_data, 'cannot access train data as attribute'
    assert config.train_data.dataset, 'cannot access grandchild attribute'
    assert isinstance(config.train_data.dataset.transform_pipeline[0], dict), \
        f'expected attribute transform_pipeline elements to be a dict, but found {type(config.train_data.dataset.transform_pipeline)}'


@pytest.mark.cpu
def test_load_ophooks():
    dict = {'type': 'MemTracerOpHook', 'warmup': 10, 'refreshrate': 20}
    ophook = build_ophooks(dict)
    assert ophook.refreshrate == 20
    assert ophook.warmup == 10
