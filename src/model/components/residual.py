import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import warnings


class ResidualBlock(nn.Module):
    """
    The building blocks of ResNets
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or stride != 1:
            self.conv3 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResidualBlockTranspose(nn.Module):
    """
    A transpose version of ResidualBlock
    """

    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        if stride != 3:
            warnings.warn(
                f"A stride of {stride} used with a transpose convolution with kernel size 3 will produce checkerboard patterns. These can be overcome with more training, but may be detrimental"
            )
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            output_padding=output_padding,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or stride != 1:
            self.conv3 = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                output_padding=output_padding,
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResidualBlockResize(nn.Module):
    """
    The building blocks of ResNets using bilinear resizing instead of convolutions for change in size.
    N.B: there is currently no determinstic backward function for bilinear upsample and therefore this cannot be used with determinism.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        factor=1,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.factor = factor
        self.interpolation = interpolation

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels or factor != 1:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = transforms.functional.resize(
            Y,
            [int(s * self.factor) for s in Y.shape[-2:]],
            interpolation=self.interpolation,
        )
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
            X = transforms.functional.resize(
                X,
                [int(s * self.factor) for s in X.shape[-2:]],
                interpolation=self.interpolation,
            )
        Y += X
        return F.relu(Y)
