#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

DEPTH = 2
BATCH_SIZE = 512
SEQ_LENGTH = 128
HIDDEN_SIZE = 512
NUM_CLASSES = 1000
NUM_BLOCKS = 6
IMG_SIZE = 224

def check_equal(A, B):
    return torch.allclose(A, B, rtol=1e-4, atol=1e-2)
