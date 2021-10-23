"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
from torch.functional import split
import torch.nn as nn
from collections import namedtuple

ConvBlock = namedtuple("ConvBlock", "kernel_size filters stride padding")

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""
architecture_config = [
    ConvBlock(7, 64, 2, 3),
    "M",
    ConvBlock(3, 192, 1, 1),
    "M",
    ConvBlock(1, 128, 1, 0),
    ConvBlock(3, 256, 1, 1),
    ConvBlock(1, 256, 1, 0),
    ConvBlock(3, 512, 1, 1),
    "M",
    [ConvBlock(1, 256, 1, 0), ConvBlock(3, 512, 1, 1), 4],
    ConvBlock(1, 512, 1, 0),
    ConvBlock(3, 1024, 1, 1),
    "M",
    [ConvBlock(1, 512, 1, 0), ConvBlock(3, 1024, 1, 1), 2],
    ConvBlock(3, 1024, 1, 1),
    ConvBlock(3, 1024, 2, 1),
    ConvBlock(3, 1024, 1, 1),
    ConvBlock(3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels, split_size, num_boxes, num_classes):
        super().__init__()

        self.architecture = architecture_config
        self.in_channels = in_channels

        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)

    def forward(self, x):
        x = self.darknet(x)

        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_fcs(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, split_size * split_size *
                      (num_classes + num_boxes * 5)),
        )

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            # CNN Block
            if type(x) == ConvBlock:
                layers.append(
                    CNNBlock(in_channels, x.filters,
                             x.kernel_size, x.stride, x.padding)
                )
                in_channels = x.filters
            # Maxpolling with kernel 2 stride 2
            elif type(x) == str:
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            # repeat CNN Blocks
            elif type(x) == list:
                convs = x[0: -1]
                num_repeats = x[-1]

                for _ in range(num_repeats):
                    for x in convs:
                        layers.append(
                            CNNBlock(in_channels, x.filters,
                                     x.kernel_size, x.stride, x.padding)
                        )
                        in_channels = x.filters

        return nn.Sequential(*layers)


def test_model_output_shape():
    split_size = 7
    num_anchors = 2
    num_classes = 10
    net = Yolov1(3, split_size, num_anchors, num_classes)

    x = torch.rand((1, 3, 448, 448))

    output = net(x)

    assert output.ndim == 2
    assert output.shape[1] == split_size ** 2 * (num_classes + num_anchors * 5)
