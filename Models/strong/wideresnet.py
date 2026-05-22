import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    WideResNet for CIFAR-10.
    Default: depth=28, widen_factor=10 -> WRN-28-10
    You can reduce widen_factor to 2 or 4 for lighter experiments.
    """
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 2,
        dropout: float = 0.3,
        num_classes: int = 10,
    ):
        super().__init__()

        if (depth - 4) % 6 != 0:
            raise ValueError("depth should satisfy (depth - 4) % 6 == 0")

        n = (depth - 4) // 6
        k = widen_factor

        widths = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(
            3, widths[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        self.block1 = self._make_layer(
            widths[0], widths[1], n, stride=1, dropout=dropout
        )
        self.block2 = self._make_layer(
            widths[1], widths[2], n, stride=2, dropout=dropout
        )
        self.block3 = self._make_layer(
            widths[2], widths[3], n, stride=2, dropout=dropout
        )

        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        dropout: float,
    ):
        layers = [WideBasicBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(WideBasicBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def WRN_28_2(num_classes: int = 10, dropout: float = 0.3) -> WideResNet:
    return WideResNet(depth=28, widen_factor=2, dropout=dropout, num_classes=num_classes)


def WRN_28_4(num_classes: int = 10, dropout: float = 0.3) -> WideResNet:
    return WideResNet(depth=28, widen_factor=4, dropout=dropout, num_classes=num_classes)