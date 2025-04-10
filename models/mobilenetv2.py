import torch
import torch.nn as nn
from .blocks import InvertedResidual


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0):
        super().__init__()
        # Config: t, c, n, s
        # t: expansion factor
        # c: output channels
        # n: number of times repeated
        # s: stride (only first block in group uses it)
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # originally s=2, but CIFAR doesn't need spatial downscaling here
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )

        layers = []
        for t, c, n, s in self.cfgs:
            out_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, out_channel, stride, t))
                input_channel = out_channel
        self.features = nn.Sequential(*layers)

        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x
