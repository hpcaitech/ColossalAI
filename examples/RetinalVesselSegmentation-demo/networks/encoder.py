import math
import torch
import torch.nn as nn

class EncoderDC(nn.Module):
    def __init__(self, Num_D, backbone, BatchNorm):
        super(EncoderDC, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        inplanes = 256
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.cls = nn.Conv2d(inplanes, Num_D, 1)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls(x)

        return torch.squeeze(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_encoderDC(Num_D, backbone, BatchNorm):
    return EncoderDC(Num_D, backbone, BatchNorm)
