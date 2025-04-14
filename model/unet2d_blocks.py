# building blocks

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A residual block with 2 convolutional layers and skip connection.
    It does not downsample the input.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=None,  # Changed to None so we can auto-calculate
        norm_layer=nn.GroupNorm,
        activation=nn.GELU,
        bias=True,
        groups=1,
    ):
        super().__init__()
        # Auto-calculate padding to preserve spatial dimensions
        if padding is None:
            # For odd kernel sizes, integer division works
            assert kernel_size % 2 == 1, "kernel_size must be odd to auto-calculate padding"
            padding = kernel_size // 2
            
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
            norm_layer(groups, out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias),
            norm_layer(groups, out_channels),
            activation(),
        )
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.norm_layer = norm_layer(groups, out_channels)
        self.activation = activation()

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.layers(x)
        out += identity
        out = self.norm_layer(out)
        out = self.activation(out)
        return out
    
class TimeEmbedding(nn.Module):
    """
    A module that embeds the time step into the input tensor.
    It uses a linear layer followed by a SILU activation and a normalization layer.
    """
    def __init__(
        self,
        time_emb_dim,
    ):
        super().__init__()
        assert time_emb_dim % 2 == 0, "time_emb_dim is expected to be even"
        self.time_emb_dim = time_emb_dim

    def forward(self, t):
        # Sinusoidal position embedding
        half_dim = self.time_emb_dim // 2
        index = torch.arange(half_dim, dtype=torch.float32, device=t.device)
        index = 10000 ** (2 * index / half_dim)
        time = t[:, None] / index[None, :]
        time = torch.cat((time.sin(), time.cos()), dim=-1)
        time = time.view(time.shape[0], self.time_emb_dim)
        return time
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=num_heads, batch_first=True
        )
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        
        qkv = self.qkv(x).reshape(b, 3, c, h*w).permute(1, 0, 3, 2)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, h*w, c]
        
        x, _ = self.attention(q, k, v)        
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj(x)
        
        return x + residual
    
class DownsampleBlock(nn.Module):
    """
    A downsampling block that contains ResidualBlock, self-attention, and pooling layers.
    """
    def __init__(
        self,
        in_channels,
        out_channels, # suppose to be 2 * in_channels
        groups=16,
        time_emb_dim=128,
    ):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, in_channels)
        self.attention = MultiHeadAttentionBlock(in_channels, in_channels)
        self.block2 = ResidualBlock(in_channels, in_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2) # downsample
        self.skip_connection = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm_layer = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()

        self.time_emb = TimeEmbedding(time_emb_dim)
        self.time_emb_transform = nn.Sequential(
            nn.Linear(time_emb_dim, in_channels),
            nn.SiLU(),
        )


    def forward(self, x, t):
        # time embedding
        time_emb = self.time_emb(t)
        time_emb = self.time_emb_transform(time_emb)
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
        time_emb = time_emb.expand(-1, -1, x.shape[2], x.shape[3])

        identity = self.skip_connection(x)
        out = self.block1(x)
        out = out + time_emb
        out = self.attention(out)
        out = self.block2(out)
        out += identity
        before_pool = out
        out = self.downsample(out)
        out = self.norm_layer(out)
        out = self.activation(out)
        return out, before_pool

class UpsampleBlock(nn.Module):
    """
    An upsampling block that contains ResidualBlock, self-attention, and upsampling layers.
    """
    def __init__(
        self,
        in_channels,
        out_channels, # suppose to be in_channels // 2
        additional_channels, # for skip connection from downsample blocks
        groups=16,
        time_emb_dim=128,
    ):
        super().__init__()
        # self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) 
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )       
        # Merge skip connections
        self.merge = nn.Conv2d(out_channels + additional_channels, out_channels, kernel_size=1)        
        self.block1 = ResidualBlock(out_channels, out_channels)
        self.attention = MultiHeadAttentionBlock(out_channels, out_channels)
        self.block2 = ResidualBlock(out_channels, out_channels)        
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.time_emb_transform = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
        )        
        self.skip_connection = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm_layer = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()

    def forward(self, x, skip_features, t):
        out = self.upsample(x)

        time_emb = self.time_emb(t)
        time_emb = self.time_emb_transform(time_emb)
        time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
        time_emb = time_emb.expand(-1, -1, out.shape[2], out.shape[3])  
        
        out = torch.cat([out, skip_features], dim=1)
        out = self.merge(out)
        identity = self.skip_connection(out)
        out = self.block1(out)
        out = out + time_emb
        out = self.attention(out)
        out = self.block2(out)
        out += identity
        out = self.norm_layer(out)
        out = self.activation(out)
        return out
    
class BottleneckBlock(nn.Module):
    """
    A bottleneck block that contains ResidualBlock and self-attention layers.
    The size is not changed.
    """
    def __init__(self, in_channels, groups=16, time_emb_dim=128):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, in_channels)
        self.attention = MultiHeadAttentionBlock(in_channels, in_channels)
        self.block2 = ResidualBlock(in_channels, in_channels)
        
        # Add time embedding
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, in_channels),
            nn.SiLU(),
        )
        
    def forward(self, x, t):
        # Add time conditioning
        time_emb = self.time_emb(t)
        time_emb = self.time_proj(time_emb)
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)
        
        identity = x
        out = self.block1(x)
        out = out + time_emb  # Add time embedding
        out = self.attention(out)
        out = self.block2(out)
        out = out + identity
        return out

