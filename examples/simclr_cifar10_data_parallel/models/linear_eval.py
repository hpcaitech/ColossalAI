import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbone import backbone

class Linear_eval(nn.Module):

    def __init__(self, model='resnet18', class_num=10, **kwargs):
        super().__init__()
        
        self.backbone = backbone(model, **kwargs)
        self.backbone.requires_grad_(False)
        self.fc = nn.Linear(self.backbone.output_dim, class_num)
        
    def forward(self, x):

        out = self.backbone(x)
        out = self.fc(out)
        return out
