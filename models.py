import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool0 = nn.MaxPool2d(2)  # 512 -> 256
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # 256 -> 128
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)  # 128 -> 64
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)  # 64 -> 32

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # decoder (upsampling)
        self.upsample0 = nn.UpsamplingBilinear2d(scale_factor=2)  # 32 -> 64
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)  # 64 -> 128
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)  # 128 -> 256
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)  # 256 -> 512
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
        )

    def forward(self, x):
        e00 = self.enc_conv0(x)
        e0 = self.pool0(e00)
        e11 = self.enc_conv1(e0)
        e1 = self.pool1(e11)
        e22 = self.enc_conv2(e1)
        e2 = self.pool2(e22)
        e33 = self.enc_conv3(e2)
        e3 = self.pool3(e33)

        # bottleneck
        b = self.bottleneck_conv(e3)

        # decoder
        d0 = self.upsample0(b)
        d0 = torch.cat((d0, e33), dim=1)
        d0 = self.dec_conv0(d0)
        d1 = self.upsample1(d0)
        d1 = torch.cat((d1, e22), dim=1)
        d1 = self.dec_conv1(d1)
        d2 = self.upsample2(d1)
        d2 = torch.cat((d2, e11), dim=1)
        d2 = self.dec_conv2(d2)
        d3 = self.upsample3(d2)
        d3 = torch.cat((d3, e00), dim=1)
        d3 = self.dec_conv3(d3)

        return d3


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # we stack images! input has 6 channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 512 -> 256
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # we stack images! input has 6 channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),  # we stack images! input has 6 channels
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.final = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 8 -> 4
        )

    def forward(self, x):

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final(x)
        x = torch.sigmoid(torch.mean(x))
        return x

