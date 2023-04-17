import torch
from torch import nn

from conv_block import ConvBlock
from srgan_residual_block import SRGanResidualBlock
from upsample_block import UpsampleBlock


class Generator(nn.Module):
    def __init__(self, num_residuals=16, embed_size=64, total_upscale=4):
        super().__init__()
        assert (total_upscale & (total_upscale - 1)) == 0, "Only supports up-scaling in powers of 2 (2x, 4x, etc)"

        self.conv_before_residuals = ConvBlock(in_channels=3,  # Assume 3-channel RGB image
                                               out_channels=embed_size,
                                               kernel_size=9,
                                               stride=1,
                                               padding=4,  # same padding
                                               use_bn=False,
                                               use_act=True,
                                               is_disc=False)

        self.residuals = nn.Sequential(*[SRGanResidualBlock(in_channels=embed_size) for _ in range(num_residuals)])

        self.conv_after_residuals = ConvBlock(in_channels=embed_size,
                                              out_channels=embed_size,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,  # same padding
                                              use_bn=True,
                                              use_act=False,
                                              is_disc=False)

        num_upsamples = int(torch.log2(torch.tensor(total_upscale)).item())
        self.upsamples = nn.Sequential(
            *[UpsampleBlock(in_channels=embed_size,
                            scale_factor=2)
              for _ in range(num_upsamples)]
        )

        self.conv_after_upsample = nn.Conv2d(in_channels=embed_size,
                                             out_channels=3,  # map back to 3-channel RGB image
                                             kernel_size=9,
                                             stride=1,
                                             padding=4)  # same padding

    def forward(self, x):
        # input shape: (B, 3, H, W)
        x = self.conv_before_residuals(x)  # (B, EMBED_SIZE, H, W)
        y = self.residuals(x)  # (B, EMBED_SIZE, H, W)
        y = self.conv_after_residuals(y)  # (B, EMBED_SIZE, H, W)
        y = y + x  # (B, EMBED_SIZE, H, W)
        y = self.upsamples(y)  # (B, EMBED_SIZE, H * TOTAL_UPSCALE, W * TOTAL_UPSCALE)
        y = self.conv_after_upsample(y)  # (B, 3, H * TOTAL_UPSCALE, W * TOTAL_UPSCALE)
        return y
