import numpy as np
from torch import nn


class ConvBlock(nn.Module):
    """
    Convolutional block used in SRGAN. Consists of:
        1. convolutional layer
        2. (optional) batch norm layer
        3. PReLu (in generator) OR LeakyReLU (in discriminator)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 use_bn, use_act, is_disc, leaky_slope=0.2):
        super().__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels) if use_bn else nn.Identity()
        self.relu = nn.LeakyReLU(leaky_slope) if is_disc else nn.PReLU(num_parameters=out_channels)

    def forward(self, x):
        # input shape: (B, in_C, in_H, in_W)
        x = self.conv(x)  # (B, out_C, out_H, out_W)
        x = self.bn(x)  # (B, out_C, out_H, out_W)
        if self.use_act:
            x = self.relu(x)  # (B, out_C, out_H, out_W)
        return x
