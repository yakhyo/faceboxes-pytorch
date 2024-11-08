import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Union, Optional, Tuple


class ConvBlockBase(nn.Module):
    """Base class for Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            inplace: bool = True,
            bias: bool = False,
            concat_before_relu: bool = False
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels, eps=1e-5)
        self.relu = nn.ReLU(inplace=inplace)
        self.concat_before_relu = concat_before_relu

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        if self.concat_before_relu:
            x = torch.cat([x, -x], 1)  # concatenate inputs for ReLU
        x = self.relu(x)
        return x


class ConvBNReLU(ConvBlockBase):
    """Standard Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            inplace: bool = True,
            bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            inplace=inplace,
            bias=bias,
            concat_before_relu=False
        )


class CReLU(ConvBlockBase):
    """Standard Convolutional Block with additional concat before ReLU function"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            inplace: bool = True,
            bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            inplace=inplace,
            bias=bias,
            concat_before_relu=True
        )


class Inception(nn.Module):
    """Inception block"""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch1x1 = ConvBNReLU(in_channels=in_channels, out_channels=32, kernel_size=1)
        self.branch1x1_2 = ConvBNReLU(in_channels=in_channels, out_channels=32, kernel_size=1)

        self.branch3x3_reduce = ConvBNReLU(in_channels=in_channels, out_channels=24, kernel_size=1)
        self.branch3x3 = ConvBNReLU(in_channels=24, out_channels=32, kernel_size=3)

        self.branch3x3_reduce_2 = ConvBNReLU(in_channels=in_channels, out_channels=24, kernel_size=1)
        self.branch3x3_2 = ConvBNReLU(in_channels=24, out_channels=32, kernel_size=3)
        self.branch3x3_3 = ConvBNReLU(in_channels=32, out_channels=32, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)

        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)

        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        return torch.cat(outputs, 1)


class FeatureExtractor(nn.Module):
    """Rapidly Digested and Multiple Scale Convolutional Layers (https://arxiv.org/abs/1708.05234)"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        filters = [3, 24, 48, 64, 128, 256]
        self.num_classes = num_classes

        self.conv1 = CReLU(in_channels=filters[0], out_channels=filters[1], kernel_size=7, stride=4)
        self.conv2 = CReLU(in_channels=filters[2], out_channels=filters[3], kernel_size=5, stride=2)

        self.inception1 = Inception(in_channels=128)
        self.inception2 = Inception(in_channels=128)
        self.inception3 = Inception(in_channels=128)

        self.conv3_1 = ConvBNReLU(filters[4], filters[4], kernel_size=1, stride=1, padding=0)
        self.conv3_2 = ConvBNReLU(filters[4], filters[5], kernel_size=3, stride=2, padding=1)

        self.conv4_1 = ConvBNReLU(filters[5], filters[4], kernel_size=1, stride=1, padding=0)
        self.conv4_2 = ConvBNReLU(filters[4], filters[5], kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.inception1(x)
        x = self.inception2(x)
        scale1 = self.inception3(x)  # [batch_size, 128, 32, 32]

        x = self.conv3_1(scale1)
        scale2 = self.conv3_2(x)  # [batch_size, 256, 16, 16]

        x = self.conv4_1(scale2)
        scale3 = self.conv4_2(x)  # [batch_size, 256, 8, 8]

        return scale1, scale2, scale3


class FaceBoxes(nn.Module):
    """FaceBoxes Module"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = FeatureExtractor(num_classes)

        anchors = [21, 1, 1]
        filters = [128, 256, 256]

        self.loc_layers = nn.Sequential(
            nn.Conv2d(filters[0], anchors[0] * 4, kernel_size=3, padding=1),
            nn.Conv2d(filters[1], anchors[1] * 4, kernel_size=3, padding=1),
            nn.Conv2d(filters[2], anchors[2] * 4, kernel_size=3, padding=1)
        )

        self.conf_layers = nn.Sequential(
            nn.Conv2d(filters[0], anchors[0] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(filters[1], anchors[1] * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(filters[2], anchors[2] * num_classes, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        loc = []
        conf = []

        # multi scale features
        features: Tuple[Tensor, Tensor, Tensor] = self.feature_extractor(x)

        for (f, l, c) in zip(features, self.loc_layers, self.conf_layers):
            loc.append(l(f).permute(0, 2, 3, 1).contiguous())
            conf.append(c(f).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([l.view(l.size(0), -1) for l in loc], 1)
        conf = torch.cat([c.view(c.size(0), -1) for c in conf], 1)

        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, 2)

        if self.training:
            return loc, conf
        return loc, F.softmax(conf, dim=-1)
