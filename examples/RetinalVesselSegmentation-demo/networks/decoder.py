import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.MetaEmbedding import MetaEmbedding


class Decoder(nn.Module):
    def __init__(self, num_classes, num_domain, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn = BatchNorm(304)
        self.relu = nn.ReLU()
        self.embedding = MetaEmbedding(304, num_domain)
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, feature, low_level_feat, domain_code, centroids):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat_ = low_level_feat
        # x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = F.interpolate(feature, size=low_level_feat_.size()[2:], mode='bilinear', align_corners=True)
        feature = torch.cat((x, low_level_feat_), dim=1)
        feature = self.bn(feature)
        # feature = self.relu(feature)
        x, hal_scale, sel_scale = self.embedding(feature, domain_code, centroids)

        x = self.last_conv(x)

        return x, feature, hal_scale, sel_scale

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, num_domain, backbone, BatchNorm):
    return Decoder(num_classes, num_domain, backbone, BatchNorm)
