#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

DEPTH = 2
BATCH_SIZE = 512
SEQ_LENGTH = 128
HIDDEN_SIZE = 768
NUM_CLASSES = 1000
NUM_BLOCKS = 6
IMG_SIZE = 224

def check_equal(A, B):
    eq = torch.allclose(A, B, rtol=1e-3, atol=1e-2)
    assert eq
    return eq
