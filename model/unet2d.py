# a UNet implmentation with same input size and output size
# inconv->down1->down2->down3->bottleneck->up3->up2->up1->outconv

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet2d_blocks import (
    ResidualBlock,
    DownsampleBlock,
    UpsampleBlock,
)

class SimpleUNet2D(nn.Module):
    def __init__(
        self,
        num_blocks=4,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        time_emb_dim=128,
        groups=16,
    ):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        ch_arrange = []
        for i in range(num_blocks):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            ch_arrange.append((in_ch, out_ch))
        
        for in_ch, out_ch in ch_arrange:
            self.down_blocks.append(
                DownsampleBlock(in_ch, out_ch, groups=groups, time_emb_dim=time_emb_dim)
            )
        for in_ch, out_ch in reversed(ch_arrange):
            self.up_blocks.append(
                UpsampleBlock(out_ch, in_ch, in_ch, groups=groups, time_emb_dim=time_emb_dim)
            )
        
        self.bottleneck = ResidualBlock(base_channels * (2 ** num_blocks), base_channels * (2 ** num_blocks))
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        x = self.in_conv(x)
        
        # downsampling
        skip_connections = []
        for i, down_block in enumerate(self.down_blocks):
            x, before_pool = down_block(x, t)
            skip_connections.append(before_pool)

        # bottleneck
        x = self.bottleneck(x)

        # upsampling
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_connections[-(i + 1)], t)

        x = self.out_conv(x)
        
        return x