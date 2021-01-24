import torch
import torch.nn as nn
import torch.nn.functional as F


class TasselNetv2(nn.Module):
    # replace the first fully-connected layer with avgpool
    # change the position of maxpool3
    def __init__(self,bn=True,in_channel=1):
        super(TasselNetv2, self).__init__()
        self.bn = bn
        self.in_channel = in_channel
        self.rf = 32
        if bn:
            self.layer1  = nn.Sequential(
                nn.Conv2d(in_channel, 16, 3, padding = 1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            self.layer2  = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding = 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.layer3  = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding = 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )


            self.pool1   = nn.MaxPool2d((2, 2), stride=2)
            self.pool2   = nn.MaxPool2d((2, 2), stride=2)
            self.pool3   = nn.MaxPool2d((2, 2), stride=2)

            self.avgpool = nn.AvgPool2d((4, 4), stride=4)

            self.predict = nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
                nn.Softplus()
            )
        else:
            self.layer1  = nn.Sequential(
                nn.Conv2d(in_channel, 16, 3, padding = 1),
                nn.ReLU(inplace=True)
            )
            self.layer2  = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding = 1),
                nn.ReLU(inplace=True)
            )
            self.layer3  = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding = 1),
                nn.ReLU(inplace=True)
            )


            self.pool1   = nn.MaxPool2d((2, 2), stride=2)
            self.pool2   = nn.MaxPool2d((2, 2), stride=2)
            self.pool3   = nn.MaxPool2d((2, 2), stride=2)

            self.avgpool = nn.AvgPool2d((4, 4), stride=4)

            self.predict = nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
                nn.Softplus()
            )      

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.avgpool(x)
        x = self.predict(x)
        return x
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                        m.weight, 
                        mode='fan_in', 
                        nonlinearity='relu'
                        )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, relu=True, bn=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x