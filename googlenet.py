import torch
import torch.nn as nn


class GoogLeNet(nn.module):
    def __init__(self, img_in_channels=3, num_classes=1000):
        super().__init__()
        self.conv1 = ConvBlock(img_in_channels, 64, kernel_size=7, stride=2, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320,  32, 128, 128)

        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.max_pool(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        x = self.max_pool_4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class Inception:
    def __init__(self, in_channels, out_1_1, reduce_3_3, out_3_3, reduce_5_5, out_5_5, out_1_1_pool):
        super().__init__(self)
        self.branch1 = conv_block(in_channels, out_1_1, kernel_size=1)
        self.branch2 = nn.sequential(ConvBlock(in_channels,reduce_3_3,kernel_size=1),
                                     ConvBlock(reduce_3_3, out_3_3,kernel_size=3,padding=1))
        self.branch3 = nn.sequential(ConvBlock(in_channels, reduce_5_5, kernel_size=1),
                                     ConvBlock(in_channels, out_5_5, kernel_size=5, padding=2))

        self.branch4 = nn.sequential(nn.MaxPool2d(kernel_size=3, padding=1),
                                     ConvBlock(in_channels, out_1_1_pool, kernel_size=5, padding=2))

    def forward(self, x):
        return torch.cat([self.branch1 + self.branch2 + self.branch3 + self.branch4], 1)


class ConvBlock:
    def __init__(self, in_channels,out_channels, **kwargs):
        super().__init__(self)
        self.conv1 = nn.Conv2d(in_channels, out_channels,**kwargs)
        self.bn1 =nn.BatchNorm2D(out_channels)
        self.relu = nn.Relu()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))

if __name__ == '__main__':
    x = torch.rand(3,3,224,224)
    model = GoogLeNet()
    print(model(x).shape)