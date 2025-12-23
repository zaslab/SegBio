# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 17:07:13 2025

@author: eduar
"""

# Parameterized UNet (variable depth)
#
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:  
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        return self.block(x)


def Norm(num_channels: int, *, max_groups: int = 8) -> nn.GroupNorm:
        """
        GroupNorm is robust for small batch sizes.
    
        We pick the largest number of groups <= max_groups that divides num_channels.
        """
        g = min(max_groups, num_channels)
        while g > 1 and (num_channels % g) != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g, num_channels=num_channels)

class UNet(nn.Module):
    """
    UNet with configurable depth.
      - depth: number of encoder levels (downsamplings). Typical: 3–5.
      - base_filters: channels at the first level; each deeper level doubles it.
      - Uses ConvTranspose2d upsampling; pads to handle odd sizes.
    Returns LOGITS (no sigmoid).
    """
    def __init__(self,
                 n_channels: int = 1,
                 n_classes: int = 3,
                 base_filters: int = 32,
                 depth: int = 3):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        # Channel plan per encoder stage
        chans = [base_filters * (2 ** i) for i in range(depth)]  # e.g., [32,64,128] for depth=3
        self.depth = depth

        # Encoder
        self.enc0 = DoubleConv(n_channels, chans[0])
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(chans[i-1], chans[i]),
            ))

        # Bottleneck (one level deeper, 2× channels of the deepest encoder)
        self.bottleneck = DoubleConv(chans[-1], chans[-1] * 2)

        # Decoder (upsample + concat skip + DoubleConv) — one block per encoder level (reversed)
        self.upconvs   = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        in_ch = chans[-1] * 2
        for skip_ch in reversed(chans):
            self.upconvs.append(nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(skip_ch * 2, skip_ch))
            in_ch = skip_ch

        # Output head
        self.out_conv = nn.Conv2d(chans[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (store skips)
        skips = []
        x = self.enc0(x)      # level 0
        skips.append(x)
        for down in self.downs:
            x = down(x)       # levels 1..depth-1
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (mirror)
        for i in range(self.depth):  # i = 0..depth-1
            x_up = self.upconvs[i](x)
            # Match spatial size to the corresponding skip (pad if needed)
            skip = skips[-(i+1)]  # last skip first
            diff_y = skip.size(2) - x_up.size(2)
            diff_x = skip.size(3) - x_up.size(3)
            if diff_y or diff_x:
                x_up = F.pad(x_up, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x_up], dim=1)
            x = self.decoders[i](x)

        return self.out_conv(x)   # logits (apply sigmoid/softmax outside if needed)
def load_unet(
    ckpt: Path,
    device: torch.device = "cpu",
    *,
    n_channels: int = 1,
    n_classes: int = 3,
    base_filters: int = 32,
    depth: int = 4
) -> UNet:
    """Instantiate **UNet3Layer** and load a *state_dict* from *ckpt*.

    Parameters
    ----------
    ckpt
        Path to a ``.pth`` file saved via ``torch.save(model.state_dict())``.
    device
        "cpu", "cuda", or a *torch.device*.
    base_filters
        Must match the setting used during training or loading will fail.
    """
    device = torch.device(device)
    model = UNet(
        n_channels=n_channels,
        n_classes=n_classes,
        base_filters=base_filters,
        depth=depth
        
    ).to(device)
    state_dict: dict[str, Any] = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model