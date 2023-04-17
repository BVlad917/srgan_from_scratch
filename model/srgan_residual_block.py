from torch import nn

from conv_block import ConvBlock


class SRGanResidualBlock(nn.Module):
    """
    Residual Block used in SRGAN. Consists of:
        1. Conv Block which keeps the same number of channels (also keep the H/W the same as in the paper); has
        BatchNorm and PReLu
        2. Conv Block which keeps the same number of channels (also keep the H/W the same as in the paper), has
        BatchNorm but no activation
        3. Residual connection with the input
    """

    def __init__(self, in_channels):
        super().__init__()
        # 2 conv blocks in sequence; assumes same padding so only channels dimension will change
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               use_bn=True,
                               use_act=True,
                               is_disc=False)
        self.conv2 = ConvBlock(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               use_bn=True,
                               use_act=False,
                               is_disc=False)

    def forward(self, x):
        # input shape: (B, in_C, H, W)
        y = self.conv1(x)  # (B, in_C, H, W)
        y = self.conv2(y)  # (B, in_C, H, W)
        return y + x
