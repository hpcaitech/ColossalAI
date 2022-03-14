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

    detector = TensorDetector(log='test', include_cpu=False, module=model)

    detector.detect()
    out = model(data)

    detector.detect()
    loss = out.sum()
    detector.detect()
    loss.backward()
    detector.detect()
    model = MLP().cuda()
    detector.detect()
    detector.close()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    test_tensor_detect()