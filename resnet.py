import torch
import torch.nn as nn


class Resnet:

    def __init__(self, img_in_channels, num_layers_list, num_classes):
        super(Resnet, self).__init__()
        self.in_chans = 64
        self.conv1 = Conv2d(img_in_channels, out_chan=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Relu()
        self.maxpool = Maxpool2d(kernel=3, stride=2, padding=1)

        self.layer_block_1 = self._make_layer_block(num_layers_list[0], out_chan=64, stride=1)
        self.layer_block_2 = self._make_layer_block(num_layers_list[1], out_chan=128, stride=2)
        self.layer_block_3 = self._make_layer_block(num_layers_list[2], out_chan=256, stride=2)
        self.layer_block_4 = self._make_layer_block(num_layers_list[3], out_chan=512, stride=2)

        self.avgpool = AvgPool2d(1,1)
        self.fc = Linear(512*4, num_classes)
        self.softmax = Softmax()

    def _make_layer_block(self, num_residual_layers, out_chan, stride):
        identity_down_sample = None
        layers = []

        if stride !=1 | self.in_chans != out_chan*4:
            identity_down_sample = nn.sequential(nn.Conv2d(self.in_chans, out_chan*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_chan*4))
        layers.append(ResnetBlock(self.in_chans, out_chan, identity_down_sample, stride))
        self.in_chans = out_chan*4

        for i in range(num_residual_layers-1):
            layers.append(ResnetBlock(self.in_chans, out_chan))

        return nn.sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer_block_1(x)
        x = self.layer_block_2(x)
        x = self.layer_block_3(x)
        x = self.layer_block_4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResnetBlock:
    def __init__(self, in_chan, out_chan, stride=1, identity_down_sample=None):
        super(ResnetBlock, self).__init__()
        self.expension = 4
        self.conv1 = Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.conv3 = Conv2d(in_chan, out_chan*self.expension, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_chan*self.expension)
        self.relu = nn.Relu()
        self.identity_down_sample = identity_down_sample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_down_sample is not None:
            identity = self.identity_down_sample(identity)

        x += identity
        x = self.relu(x)
        return x


