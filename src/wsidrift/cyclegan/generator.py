"""ResNet-based generator for CycleGAN stain transfer."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """9-block ResNet generator for 256×256 image translation.

    Args:
        n_residual: Number of residual blocks (6 for 128px, 9 for 256px).
        ngf:        Base number of generator filters.
    """

    def __init__(self, n_residual: int = 9, ngf: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # downsampling
        for mult in [1, 2]:
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True),
            ]
        # residual blocks
        for _ in range(n_residual):
            layers.append(ResidualBlock(ngf * 4))
        # upsampling
        for mult in [2, 1]:
            layers += [
                nn.ConvTranspose2d(
                    ngf * mult * 2, ngf * mult, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(ngf * mult),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
