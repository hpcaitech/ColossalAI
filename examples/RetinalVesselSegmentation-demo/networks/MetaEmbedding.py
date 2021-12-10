import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaEmbedding(nn.Module):
    
    def __init__(self, feat_dim=256, num_domain=3):
        super(MetaEmbedding, self).__init__()
        self.num_domain = num_domain
        self.hallucinator = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.selector = nn.Sequential(
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x, domain_code, centroids, *args):
        # storing direct feature
        direct_feature = x
        # hal_scale = self.hallucinator(x)
        hal_scale = torch.softmax(domain_code, -1)

        size = centroids.size()
        centroids_ = centroids.view(centroids.size(0), -1)
        memory_feature = torch.matmul(hal_scale, centroids_)

        memory_feature = memory_feature.view(x.size(0), size[1], size[2], size[3])
        sel_scale = self.selector(x)
        infused_feature = memory_feature * sel_scale
        x = direct_feature + infused_feature
        return x, hal_scale, sel_scale

def build_MetaEmbedding(feat_dim, num_domain):
    return MetaEmbedding(feat_dim, num_domain)
