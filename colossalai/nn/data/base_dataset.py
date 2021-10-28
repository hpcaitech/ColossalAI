#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from colossalai.builder import build_transform


class BaseDataset(Dataset, ABC):

    def __init__(self, transform_pipeline: list):
        transform_list = [build_transform(cfg) for cfg in transform_pipeline]
        transform = transforms.Compose(transform_list)
        self._transform_pipeline = transform
