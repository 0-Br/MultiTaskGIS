import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PretrainedConfig


class UNetOutput():
    """"""

    def __init__(self, loss, logits, labels):
        self.loss = loss
        self.logits = logits
        self.labels = labels


class UNetConfig(PretrainedConfig):
    """"""

    def __init__(
        self,
        n_channels=3,
        n_classes=8,
        n_hidden=64,
        bilinear=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.bilinear = bilinear


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """"""
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """"""
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """"""
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False), # downsample x2
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        """"""
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """"""
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        """x2 from skip connections"""
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """"""

    def __init__(self, config: UNetConfig):
        """"""
        super(UNet, self).__init__()
        self.n_channels = config.n_channels
        self.n_hidden = config.n_hidden
        self.n_classes = config.n_classes
        self.bilinear = config.bilinear
        factor = 2 if self.bilinear else 1

        self.inc = (ConvBlock(self.n_channels, self.n_hidden))
        self.down1 = (Down(self.n_hidden * 1, self.n_hidden * 2))
        self.down2 = (Down(self.n_hidden * 2, self.n_hidden * 4))
        self.down3 = (Down(self.n_hidden * 4, self.n_hidden * 8))
        self.down4 = (Down(self.n_hidden * 8, self.n_hidden * 16 // factor))
        self.up1 = (Up(self.n_hidden * 16, self.n_hidden * 8 // factor, self.bilinear))
        self.up2 = (Up(self.n_hidden * 8, self.n_hidden * 4 // factor, self.bilinear))
        self.up3 = (Up(self.n_hidden * 4, self.n_hidden * 2 // factor, self.bilinear))
        self.up4 = (Up(self.n_hidden * 2, self.n_hidden * 1, self.bilinear))
        self.outc = (nn.Conv2d(self.n_hidden, self.n_classes, kernel_size=1))
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, x, label):
        """"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x).transpose(1, 2).transpose(2, 3)
        loss = self.loss_fct(x.reshape(-1, self.n_classes), label.reshape(-1))

        return UNetOutput(loss, x, label)
