from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  List


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DoubleConv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv1, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            GRFB(mid_channels, out_channels, stride=1, scale=0.1, visual=12)
           

        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv1(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class BasicConv(nn.Module):

        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                     bn=True, bias=False):
            super(BasicConv, self).__init__()
            self.out_channels = out_channels
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU(inplace=True) if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x

class GRFB(nn.Module):

        def __init__(self, in_channels, out_channels, stride=1, scale=0.1, visual=12):
            super(GRFB, self).__init__()
            self.scale = scale
            self.out_channels = out_channels
            inter_planes = in_channels // 8
            self.branch0 = nn.Sequential(
                BasicConv(in_channels, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                          relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride)
            )
            self.branch1 = nn.Sequential(
                BasicConv(in_channels, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=inter_planes),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual,
                          dilation=2 * visual, relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=1)
            )
            self.branch2 = nn.Sequential(
                BasicConv(in_channels, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, groups=inter_planes),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=2 * inter_planes),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3 * visual,
                          dilation=3 * visual, relu=False),
                BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=1, stride=stride)
            )

            self.ConvLinear = BasicConv(14 * inter_planes, out_channels, kernel_size=1, stride=1, relu=False)
            self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            x0 = self.branch0(x)
            x1 = self.branch1(x)
            x2 = self.branch2(x)

            out = torch.cat((x, x0, x1, x2), 1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            out = out * self.scale + short
            out = self.relu(out)

            return out


class GRFBUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(GRFBUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return {"out": logits}
