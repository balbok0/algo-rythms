import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class LittleMoreComplicatedGenerator(nn.Module):
    def __init__(self, in_size):
        super(LittleMoreComplicatedGenerator, self).__init__()

        kernel_size = 4
        padding = 1
        stride = 2

        self.fc1 = nn.Linear(in_size, 1024 * 4 * 4)

        self.block_1_1 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                1024, 512, 4,
                stride=2,
                padding=1
            )
        )
        self.block_1_2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 512, 4,
                stride=2,
                padding=1
            )
        )
        self.block_1_3 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 512, 4,
                stride=2,
                padding=1
            )
        )
        self.block_1_4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 256, 4,
                stride=2,
                padding=1
            )
        )
        self.block_1_5 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                256, 128, 4,
                stride=2,
                padding=1
            )
        )

        self.pre_block_1_6 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.conv_1_6 = nn.ConvTranspose2d(
            128, 128, 4,
            stride=2,
            padding=1
        )

        self.pre_block_1_7 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # Small dilations
        # 256 -> 1024
        self.conv_small = nn.ConvTranspose2d(
            128, 8, (4, 4),
            dilation=(7, 1),
            stride=(4, 4),
            padding=(9, 0),
            groups=8
        )

        # Small/Medium size dilations
        # 256 -> 1024
        self.conv_small_medium = nn.ConvTranspose2d(
            128, 8, (4, 4),
            dilation=(1, 9),
            stride=(4, 4),
            padding=(0, 12),
            groups=8
        )

        # Medium
        # 256 -> 1024
        self.conv_medium = nn.ConvTranspose2d(
            128, 8, (4, 11),
            dilation=(1, 26),
            stride=(4, 3),
            padding=(0, 1),
            groups=8
        )

        # Large/Medium size dilations
        # 128 -> 1024
        self.conv_large_medium = nn.ConvTranspose2d(
            128, 8, (5, 9),
            dilation=(49, 49),
            stride=(7, 5),
            padding=(31, 2),
            groups=8
        )

        # Large size dilations
        # 128 -> 1024
        self.conv_large = nn.ConvTranspose2d(
            128, 8, (9, 11),
            dilation=(49, 74),
            stride=(5, 3),
            padding=(2, 49),
            groups=8
        )


        self.last_block = nn.Sequential(
            nn.BatchNorm2d(8 * 5),
            nn.Conv2d(8 * 5, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1024, 4, 4)

        # Run Conv blocks
        x = self.block_1_1(x)
        x = self.block_1_2(x)
        x = self.block_1_3(x)
        x = self.block_1_4(x)
        x = self.block_1_5(x)

        # Fed into all dilations, but directly only to large and large medium dilations
        x = self.pre_block_1_6(x)

        # Fed into small, small medium and medium dilations
        x_smaller = self.conv_1_6(x)
        x_smaller = self.pre_block_1_7(x_smaller)

        # Split into different dilation sizes
        x_small = self.conv_small(x_smaller)
        x_small_medium = self.conv_small_medium(x_smaller)
        x_medium = self.conv_medium(x_smaller)
        x_large_medium = self.conv_large_medium(x)
        x_large = self.conv_large(x)

        x_result = torch.cat((x_small, x_small_medium, x_medium, x_large_medium, x_large), dim=1)
        x_result = self.last_block(x_result)

        return x_result
