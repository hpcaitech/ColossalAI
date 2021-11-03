#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch

DEPTH = 4
BATCH_SIZE = 512
SEQ_LENGTH = 128
IMG_SIZE = 224
HIDDEN_SIZE = 768
NUM_CLASSES = 1000

def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1) == True
