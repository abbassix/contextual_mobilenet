import torch
import torch.nn as nn
import torch.nn.functional as F
from contextual_conv import ContextualConv2d


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise conv
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                                padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # Pointwise projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


class ContextualInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential()
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )

        self.contextual_depthwise = ContextualConv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
            bias=False,
            c_dim=hidden_dim
        )
        self.bn_dw = nn.BatchNorm2d(hidden_dim)
        self.relu_dw = nn.ReLU6(inplace=True)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expand(x) if hasattr(self, 'expand') else x
        N, C, H, W = out.shape
        context = F.adaptive_avg_pool2d(out, 1).view(N, C)  # shape: (N, C)
        out = self.contextual_depthwise(out, context)
        out = self.bn_dw(out)
        out = self.relu_dw(out)
        out = self.project(out)

        if self.use_residual:
            return x + out
        else:
            return out
