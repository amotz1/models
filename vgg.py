import torch
import torch.nn as nn


class ConvBlock:
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.Relu()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))


class VggNet(nn.module):
    def __init__(self, img_channels, num_layers):
        super(self).__init__()

        self.vgg_block_1 = self._make_vgg_block(img_channels, 64, num_layers[0])
        self.maxpool_1 = nn.Maxpool2d(stride=2, padding=2)
        self.vgg_block_2 = self._make_vgg_block(64, 128, num_layers[1])
        self.maxpool_2 = nn.Maxpool2d(stride=2, padding=2)
        self.vgg_block_3 = self._make_vgg_block(128, 256, num_layers[2])
        self.maxpool_3 = nn.Maxpool2d(stride=2, padding=2)
        self.vgg_block_4 = self._make_vgg_block(256, 512, num_layers[3])
        self.maxpool_4 = nn.Maxpool2d(stride=2, padding=2)
        self.vgg_block_5 = self._make_vgg_block(512, 512, num_layers[4])
        self.maxpool_5 = nn.Maxpool2d(stride=2, padding=2)

        self.linear_1 = nn.Linear(512, 4096)
        self.linear_2 = nn.Linear(4096, 4096)
        self.linear_3 = nn.Linear(4096, 1000)

    def forward(self, x):

        x = self.vgg_block_1(x)
        x = self.maxpool_1(x)
        x = self.vgg_block_2(x)
        x = self.maxpool_2(x)
        x = self.vgg_block_3(x)
        x = self.maxpool_3(x)
        x = self.vgg_block_4(x)
        x = self.maxpool_4(x)
        x = self.vgg_block_5(x)
        x = self.maxpool_5(x)

        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)

        return x

    def _make_vgg_block(self, in_channels, out_channels, num_layers):

        layers = [ConvBlock(in_channels, out_channels)]
        for i in range(num_layers - 1):
            if i == 1:
                layers.append(ConvBlock(out_channels, out_channels, kernel_size=1, padding=0))

            layers.append(ConvBlock(out_channels, out_channels))

        return nn.sequential(*layers)


def vgg_net():
    VggNet(3, [2, 2, 3, 3, 3])







