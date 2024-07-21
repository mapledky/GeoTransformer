import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from geotransformer.modules.geotransformer import (
    GeometricTransformer,
)

class Spatial_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Spatial_Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class LaplaceMask(nn.Module):
    def __init__(self, cfg):
        super(LaplaceMask, self).__init__()
        #self.sinkmatch = Matching(dim)
        attn_out_dim = cfg.maskformer.output_dim

        self.linear_1 = nn.Linear(attn_out_dim, attn_out_dim // 2, bias=False)
        self.linear_2 = nn.Linear(attn_out_dim // 2, attn_out_dim // 4, bias=False)

        self.linear_3 = nn.Linear(attn_out_dim // 4, attn_out_dim // 8, bias=False)
        self.linear_4 = nn.Linear(attn_out_dim // 8, 1, bias=False)

        self.attention = GeometricTransformer(
                    cfg.maskformer.input_dim,
                    cfg.maskformer.output_dim,
                    cfg.maskformer.hidden_dim,
                    cfg.maskformer.num_heads,
                    cfg.maskformer.blocks,
                    cfg.maskformer.sigma_d,
                    cfg.maskformer.sigma_a,
                    cfg.maskformer.angle_k,
                    reduction_a=cfg.maskformer.reduction_a,
                )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,ref_c, src_c, ref_f, src_f):
        #src_f B,N,C
        #ref_f B,M,C
        B,N,_ = src_f.shape
        _,M,_ = ref_f.shape
        ref_f, src_f = self.attention(
            ref_c,
            src_c,
            ref_f,
            src_f,
        )
        
        #(B,N+M,attn_ouput)
        x = torch.cat((src_f, ref_f), dim=1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        # src = x[:, :N, :]
        # ref = x[:, N:N+M, :]
        # x = src @ ref.transpose(1, 2)
        x = self.sigmoid(x)# B, N+M, 1
        
        return x.reshape(B, 1, N + M).contiguous()





