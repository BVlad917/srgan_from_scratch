from torch import nn


class UpsampleBlock(nn.Module):
    """
    Upsample block used in SRGAN. Consists of:
        1. convolutional layer to increase the channels dimension
        2. PixelShuffle layer to increase the resolution and decrease the channels dimension
        3. PReLu
    """
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        # conv layer with "same" padding and which scales the channels dimension by scale_factor^2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels * scale_factor ** 2,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale_factor)
        self.relu = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        # input shape: (B, in_C, in_H, in_W)
        x = self.conv(x)  # (B, in_C ** SCALE_FACTOR^2, in_H, in_W)
        x = self.pixel_shuffle(x)  # (B, in_C, in_H * SCALE_FACTOR, in_W * SCALE_FACTOR)
        x = self.relu(x)  # (B, in_C, in_H * SCALE_FACTOR, in_W * SCALE_FACTOR)
        return x
