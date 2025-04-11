import torch
import torch.nn as nn
import torch.nn.functional as F
from .contextual_conv_py import ContextualConv2d


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Identity()
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x += identity
        return x


class ContextualInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Identity()
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
        identity = x
        # print(x.shape)
        x = self.expand(x)
        N, C, H, W = x.shape
        context = F.adaptive_avg_pool2d(x, 1).view(N, C)  # global context vector
        # print()
        # print(x.shape)
        # print(context.shape)
        x = self.contextual_depthwise(x, context)
        x = self.bn_dw(x)
        x = self.relu_dw(x)
        x = self.project(x)
        if self.use_residual:
            x += identity
        return x
