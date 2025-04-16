# building blocks for UNet2D
# ref: https://github.com/openai/improved-diffusion

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualTimeBlock(nn.Module):
    """
    A residual block optionally takes time embedding as input in forward method
    """
    def __init__(
        self,
        in_channels:int,
        time_emb_dim:int=128,
        out_channels:int=None,
        dropout:float=0.0,
        res_conv:bool=True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout
        self.res_conv = res_conv

        # Define the layers of the residual block
        self.in_conv = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                time_emb_dim,
                out_channels,
            ),
        )
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # do skip connection
        if res_conv:
            self.skip_connect = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connect = nn.Identity()

    def forward(self, x, time_emb=None):
        """
        :param x: Input tensor [BxCxHxW]
        :param time_emb: Time embedding tensor [BxT]
        :return: Output tensor [BxCxHxW]
        """
        h = self.in_conv(x)
        emb_out = self.emb_layers(time_emb).unsqueeze(2).unsqueeze(3)

        h = h + emb_out
        h = self.out_conv(h)

        return self.skip_connect(x) + h
    
class QKVAttention(nn.Module):
    def forward(self, qkv):
        """
        :param qkv: [Bx(3C)xT], C is the embedding dimension, T is the sequence length
        :return: [BxCxT]
        """
        ch = qkv.shape[1] // 3
        q, k, v = qkv.chunk(3, dim=1) # BxCxT
        
        scale = 1 / (ch ** 0.5)
        weight = torch.bmm(q.transpose(1, 2), k) * scale
        weight = F.softmax(weight, dim=-1)
        out = torch.bmm(weight, v.transpose(1, 2)).transpose(1, 2)

        return out


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        channels:int,
        num_heads:int=4,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(32, channels) 
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1) # qkv: [Bx(QKV Channel)xHxW]
        self.attention = QKVAttention() # requires compress to [Bx(3C)x(HxW)]
        self.out_conv = nn.Conv1d(channels, channels, kernel_size=1) # [BxCxHxW]

    def forward(self, x):
        b, c, h, w = x.shape
        x_skip = x.reshape(b, c, h * w)
        x = self.norm(x_skip)
        qkv = self.qkv(x)
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        attn_out = self.attention(qkv)
        attn_out = attn_out.reshape(b, c, h * w)
        attn_out = self.out_conv(attn_out)
        return (attn_out + x_skip).reshape(b, c, h, w) # [BxCxHxW]
    
class DownsampleBlock(nn.Module):
    """
    apply a conv or pooling layer to downsample the input
    """
    def __init__(self, channels, use_conv:bool=True):
        super().__init__()
        self.channels = channels
        if use_conv:
            self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        """
        :param x: Input tensor [BxCxHxW]
        :return: Output tensor [BxCx(H/2)x(W/2)]
        """
        x = self.downsample(x)
        return x

class SinusoidalPositionEmbed(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        :param x: [B] or [B,1]
        :return: [B,dim]
        """
        # Ensure input is 2D
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)  # [B] -> [B,1]
        
        b, t = x.shape
        half_dim = self.dim // 2
        emb = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * 
            -(math.log(10000.0) / half_dim)
        )
        emb = x * emb.unsqueeze(0)  # [B,1] * [1,half_dim] -> [B,half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B,dim]
        return emb  # Changed from emb.reshape(b, -1, t) to just emb