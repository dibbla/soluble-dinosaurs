# UNet2D
# ref: https://github.com/openai/improved-diffusion

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable
from model.unet2d_blocks import (
    ResidualTimeBlock,
    MultiHeadAttentionBlock,
    DownsampleBlock,
    SinusoidalPositionEmbed
)

@dataclass
class UNet2DConfig:
    in_channels:int = 3
    base_channels:int = 32
    num_res_blocks:int = 2 # number of residual blocks per downsample
    attention_layers:Iterable = (2, 4)
    dropout:float = 0.0
    channel_multiplier:Iterable = (1, 2, 4, 8) # channel multiplier for each downsample
    time_emb_dim:int = 128
    num_heads:int = 4

class SimpleUNet2D(nn.Module):
    """
    A simple & flexible UNet2D model for diffusion models
    """
    def __init__(self, config:UNet2DConfig):
        super().__init__()
        # Store config parameters
        self.config = config
        self.in_channels = config.in_channels
        self.base_channels = config.base_channels
        self.num_res_blocks = config.num_res_blocks
        self.attention_layers = config.attention_layers
        self.dropout = config.dropout
        self.channel_multiplier = config.channel_multiplier
        self.time_emb_dim = config.time_emb_dim
        self.num_heads = config.num_heads
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbed(dim=self.time_emb_dim)
        
        # Initial projection from image to feature space
        self.input_conv = nn.Conv2d(self.in_channels, self.base_channels, kernel_size=3, padding=1)
        
        # Initialize module lists for encoder, bottleneck, and decoder
        self.encoder_blocks = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Track channels for skip connections
        skip_channels = []
        ch = self.base_channels
        
        #----- ENCODER -----#
        for level, mult in enumerate(self.channel_multiplier):
            # Multiple residual blocks per level
            for _ in range(self.num_res_blocks):
                # Add residual block
                self.encoder_blocks.append(
                    ResidualTimeBlock(
                        in_channels=ch,
                        time_emb_dim=self.time_emb_dim,
                        out_channels=int(self.base_channels * mult),
                        dropout=self.dropout,
                    )
                )
                ch = int(self.base_channels * mult)
                
                # Add attention if this level should have it
                if level in self.attention_layers:
                    self.encoder_blocks.append(
                        MultiHeadAttentionBlock(
                            channels=ch,
                            num_heads=self.num_heads,
                        )
                    )
            
            # Add downsampling if not at the bottom level
            if level < len(self.channel_multiplier) - 1:
                skip_channels.append(ch)  # Record for skip connection
                self.encoder_blocks.append(DownsampleBlock(channels=ch))
        
        #----- BOTTLENECK -----#
        self.bottleneck = nn.ModuleList([
            ResidualTimeBlock(
                in_channels=ch,
                time_emb_dim=self.time_emb_dim,
                out_channels=ch,
                dropout=self.dropout,
            ),
            MultiHeadAttentionBlock(
                channels=ch,
                num_heads=self.num_heads,
            ),
            ResidualTimeBlock(
                in_channels=ch,
                time_emb_dim=self.time_emb_dim,
                out_channels=ch,
                dropout=self.dropout,
            )
        ])
        
        #----- DECODER -----#
        skip_channels = skip_channels[::-1]  # Reverse for decoder path
        for level, mult in enumerate(reversed(self.channel_multiplier)):
            # Multiple residual blocks per level plus one extra for upsampling
            for i in range(self.num_res_blocks + 1):
                # Input channels include skip connection for first block at each level
                in_ch = ch
                if i == 0 and level > 0:
                    in_ch += skip_channels.pop(0) if skip_channels else 0
                
                # Add residual block
                self.decoder_blocks.append(
                    ResidualTimeBlock(
                        in_channels=in_ch,
                        time_emb_dim=self.time_emb_dim,
                        out_channels=int(self.base_channels * mult),
                        dropout=self.dropout,
                    )
                )
                ch = int(self.base_channels * mult)
                
                # Add attention if this level should have it
                if level in self.attention_layers:
                    self.decoder_blocks.append(
                        MultiHeadAttentionBlock(
                            channels=ch,
                            num_heads=self.num_heads,
                        )
                    )
            
            # Add upsampling if not at the top level
            if level < len(self.channel_multiplier) - 1:
                self.decoder_blocks.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        
        #----- OUTPUT LAYER -----#
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        """
        :param x: Input tensor [BxCxHxW]
        :param t: Time steps tensor [B]
        :return: Output tensor [BxCxHxW]
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Encoder path (input blocks)
        h = self.input_conv(x)
        skip_connections = []
        
        for module in self.encoder_blocks:
            if isinstance(module, ResidualTimeBlock):
                h = module(h, time_emb)
            elif isinstance(module, DownsampleBlock):
                # Store activation before downsampling
                skip_connections.append(h)
                h = module(h)
            else:
                h = module(h)
        
        # Bottleneck
        for module in self.bottleneck:
            if isinstance(module, ResidualTimeBlock):
                h = module(h, time_emb)
            else:
                h = module(h)
        
        # Decoder path (output blocks)
        for module in self.decoder_blocks:
            if isinstance(module, ResidualTimeBlock):
                # Check if we need to concatenate with skip connection
                if len(skip_connections) > 0 and h.shape[2:] == skip_connections[-1].shape[2:]:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=1)
                h = module(h, time_emb)
            elif isinstance(module, nn.Upsample):
                h = module(h)
            else:
                h = module(h)
        
        # Final output projection
        return self.output_proj(h)