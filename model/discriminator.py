from torch import nn

from conv_block import ConvBlock


class Discriminator(nn.Module):
    def __init__(self, features=None):
        super().__init__()
        if features is None:
            features = [64, 64, 128, 128, 256, 256, 512, 512]  # default SRGAN features sequence

        in_channels = 3
        blocks = []
        for idx, feature in enumerate(features):
            new_conv = ConvBlock(in_channels=in_channels,  # Assume 3-channel RGB image
                                 out_channels=feature,
                                 kernel_size=3,
                                 stride=1 + (idx % 2),  # alternating stride
                                 padding=1,
                                 use_bn=(idx > 1),
                                 use_act=True,
                                 is_disc=True)
            blocks.append(new_conv)
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        # input shape: (B, 3, H, W)
        x = self.blocks(x)  # (B, 3, H // 4, W // 4)
        x = self.classifier(x)  # (B, 1)
        return x
