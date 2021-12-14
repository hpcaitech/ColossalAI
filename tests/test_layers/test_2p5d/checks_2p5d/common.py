import torch

TESSERACT_DIM = 2
TESSERACT_DEP = 2
BATCH_SIZE = 8
SEQ_LENGTH = 8
HIDDEN_SIZE = 8


def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-5, atol=1e-2) == True
