#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from colossalai.utils import TensorDetector

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64, 8),
                                 nn.ReLU(),
                                 nn.Linear(8, 32))
    
    def forward(self, x):
        return self.mlp(x)
    
def test_tensor_detect():

    data = torch.rand(64, requires_grad=True).cuda()
    data.retain_grad()
    model = MLP().cuda()

    detector = TensorDetector(False, module=model)

    detector.detect()
    out = model(data)

    detector.detect()
    loss = out.sum()
    loss.backward()
    detector.detect()

    torch.cuda.empty_cache()

if __name__ == '__main__':
    test_tensor_detect()