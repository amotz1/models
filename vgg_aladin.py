import torch
import torch.nn as nn

vggs = [64, 64, 'M', 128, 128,'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VggNet(nn.Module):
    def __init__(self, in_channels=3):
        super(VggNet, self).__init__()
        self.in_channels = in_channels
        self.conv_block = self._create_convs(vggs)
        self.fcs = nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.Linear(4096, 1000))

    def forward(self, x):
        x = self.conv_block(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x

    def _create_convs(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels), nn.ReLU()]

                in_channels = x

            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            else:
                assert False, 'unspecified type'

        return nn.Sequential(*layers)


def test():
    model = VggNet(in_channels=3)
    x = torch.rand(1, 3, 224, 224)
    print(model(x).shape)


test()