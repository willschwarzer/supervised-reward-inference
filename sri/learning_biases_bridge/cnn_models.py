from typing import Tuple

import torch
import torch.nn as nn


def _demos_to_grid(demonstrations: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert policy demonstrations (B,1,H*W,F) -> grid tensor (B,F,H,W)."""
    if demonstrations.dim() != 4:
        raise ValueError(
            f"Expected demonstrations shape (B,1,H*W,F), got {tuple(demonstrations.shape)}"
        )

    batch, n, horizon, feat_dim = demonstrations.shape
    if n != 1:
        raise ValueError(
            f"CNN/U-Net expects one policy tensor per task (n=1), got n={n}"
        )
    if horizon != height * width:
        raise ValueError(
            f"Expected horizon H*W={height*width}, got {horizon}"
        )

    x = demonstrations[:, 0].contiguous().view(batch, height, width, feat_dim)
    return x.permute(0, 3, 1, 2).contiguous()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyCNNRegressor(nn.Module):
    """Shallow convolutional policy->reward regressor."""

    def __init__(self, height: int, width: int, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.height = int(height)
        self.width = int(width)

        c = int(base_channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(inplace=True),
        )
        self.out_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, demonstrations: torch.Tensor) -> torch.Tensor:
        x = _demos_to_grid(demonstrations, self.height, self.width)
        x = self.encoder(x)
        reward_map = self.out_head(x).squeeze(1)
        return reward_map.view(reward_map.shape[0], -1)


class PolicyUNetRegressor(nn.Module):
    """Compact U-Net policy->reward regressor for spatially-structured mapping."""

    def __init__(self, height: int, width: int, in_channels: int, base_channels: int = 32):
        super().__init__()
        if (height % 4) != 0 or (width % 4) != 0:
            raise ValueError(
                f"U-Net currently expects H and W divisible by 4, got H={height}, W={width}"
            )

        self.height = int(height)
        self.width = int(width)

        c = int(base_channels)
        self.down1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 2, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.out_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, demonstrations: torch.Tensor) -> torch.Tensor:
        x = _demos_to_grid(demonstrations, self.height, self.width)

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bottleneck(self.pool2(d2))

        u2 = self.up2(b)
        x2 = torch.cat([u2, d2], dim=1)
        x2 = self.dec2(x2)

        u1 = self.up1(x2)
        x1 = torch.cat([u1, d1], dim=1)
        x1 = self.dec1(x1)

        reward_map = self.out_head(x1).squeeze(1)
        return reward_map.view(reward_map.shape[0], -1)


def infer_grid_input_shape(demos_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Utility helper mainly for testing/debugging."""
    if len(demos_shape) != 4:
        raise ValueError(f"Expected 4D demos shape, got {demos_shape}")
    _, _, horizon, feat = demos_shape
    side = int(horizon ** 0.5)
    if side * side != horizon:
        raise ValueError(f"Horizon is not a perfect square: {horizon}")
    return side, side, int(feat)
