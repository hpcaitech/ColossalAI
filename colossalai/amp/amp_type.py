#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from enum import Enum


class AMP_TYPE(Enum):
    APEX = 'apex'
    TORCH = 'torch'
    NAIVE = 'naive'
