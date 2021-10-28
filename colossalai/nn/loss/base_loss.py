#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Absctract loss class
    """

    @abstractmethod
    def calc_loss(self, *args, **kwargs):
        pass
