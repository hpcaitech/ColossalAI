import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbone import backbone

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class SimCLR(nn.Module):

    def __init__(self, model='resnet18', **kwargs):
        super().__init__()
        
        self.backbone = backbone(model, **kwargs)
        self.projector = projection_MLP(self.backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        
    def forward(self, x1, x2):

        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2