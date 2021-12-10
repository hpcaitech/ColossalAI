# camera-ready

import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, num_classes=2, is_encoder=False, is_decoder=False):
        super(AE, self).__init__()
        self.is_encoder = is_encoder
        self.is_decoder = is_decoder

        self.num_classes = num_classes
        filter_num_list = [32, 128, 128, 256, 384, 4096]

        self.conv1 = nn.Conv2d(2, filter_num_list[0], kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.norm1 = nn.GroupNorm(int(filter_num_list[0]/32), filter_num_list[0])

        self.conv2 = nn.Conv2d(filter_num_list[0], filter_num_list[1], kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(int(filter_num_list[1]/32), filter_num_list[1])

        self.conv3 = nn.Conv2d(filter_num_list[1], filter_num_list[2], kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pooling3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(int(filter_num_list[2]/32), filter_num_list[2])

        self.conv4 = nn.Conv2d(filter_num_list[2], filter_num_list[3], kernel_size=3, stride=1, padding=1, groups=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.pooling4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(int(filter_num_list[3]/32), filter_num_list[3])

        self.conv5 = nn.Conv2d(filter_num_list[3], filter_num_list[4], kernel_size=3, stride=1, padding=1, groups=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.pooling5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Linear(24576, filter_num_list[5])

        self.dconv6 = nn.ConvTranspose2d(6, 6, kernel_size=4, stride=2, padding=1, bias=False)
        self.prelu6 = nn.PReLU()
        self.conv6 = nn.Conv2d(filter_num_list[4], filter_num_list[3], kernel_size=3, stride=1, padding=1)

        self.dconv5 = nn.ConvTranspose2d(filter_num_list[3], filter_num_list[3], kernel_size=4, stride=2, padding=1, bias=False)
        self.prelu5 = nn.PReLU()
        self.conv7 = nn.Conv2d(filter_num_list[3], filter_num_list[2], kernel_size=3, stride=1, padding=1)


        self.dconv4 = nn.ConvTranspose2d(filter_num_list[2], filter_num_list[2], kernel_size=4, stride=2, padding=1, bias=False)
        self.prelu4 = nn.PReLU()
        self.conv8 = nn.Conv2d(filter_num_list[2], filter_num_list[1], kernel_size=3, stride=1, padding=1)

        self.dconv3 = nn.ConvTranspose2d(filter_num_list[1], filter_num_list[1], kernel_size=4, stride=2, padding=1, bias=False)
        self.prelu3 = nn.PReLU()
        self.conv9 = nn.Conv2d(filter_num_list[1], filter_num_list[0], kernel_size=3, stride=1, padding=1)
        self.prelu2 = nn.PReLU()
        self.conv10 = nn.Conv2d(filter_num_list[0], self.num_classes, kernel_size=3, stride=1, padding=1)

        # self.sigmoid = nn.Sigmoid()
        self._initialize_weights()



    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    # m.bias.data.copy_(1.0)
                    m.bias.data.zero_()


    def forward(self, x):

        if self.is_encoder:
            x = self.pooling1(self.conv1(x))
            x = self.pooling2(self.conv2(x))
            x = self.pooling3(self.conv3(x))
            x = self.pooling4(self.conv4(x))
            x = self.pooling5(self.conv5(x))
            x = x.view([-1, 24576])
            # x = self.fc(x)
            return x
        else:
            if self.is_decoder:
                x = self.prelu6(self.conv6(F.interpolate(x, size=(16, 16))))
                x = self.prelu5(self.conv7(F.interpolate(x, size=(32, 32))))
                x = self.prelu4(self.conv8(F.interpolate(x, size=(64, 64))))
                x = self.prelu3(self.conv9(F.interpolate(x, size=(128, 128))))
                return x
            else:
                x = self.pooling1(self.relu1(self.conv1(x)))
                x = self.pooling2(self.relu2(self.conv2(x)))
                x = self.pooling3(self.relu3(self.conv3(x)))
                x = self.pooling4(self.relu4(self.conv4(x)))
                x = self.pooling5(self.relu5(self.conv5(x)))
                x = self.prelu6(self.conv6(F.interpolate(x, size=(16, 16))))
                x = self.prelu5(self.conv7(F.interpolate(x, size=(32, 32))))
                x = self.prelu4(self.conv8(F.interpolate(x, size=(64, 64))))
                x = self.prelu3(self.conv9(F.interpolate(x, size=(128, 128))))
                x = self.prelu2(self.conv10(F.interpolate(x, size=(256, 256))))
                return x


