import torch.nn as nn
import torch.nn.functional as F


class SimpleGenerator(nn.Module):
    def __init__(self, in_size):
        super(SimpleGenerator, self).__init__()

        kernel_size = 4
        padding = 1
        stride = 2

        self.fc1 = nn.Linear(in_size, 1024 * 4 * 4)

        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                1024, 512, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 512, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 512, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                512, 256, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                256, 128, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                128, 128, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                128, 128, 4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                128, 1, 4,
                stride=2,
                padding=(2, 1),
                dilation=(2, 1)
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1024, 4, 4)
        x = self.net(x)

        return x


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()

        self.bn0 = nn.BatchNorm2d(1)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(128, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.fc_last = nn.Linear(512, 2)

    def forward(self, x):
        x = self.bn0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = x.view(len(x), -1)

        x = self.fc_last(x)
        return x
