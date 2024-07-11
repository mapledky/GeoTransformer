import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from geotransformer.modules.mask.learn_sinkhorn import Matching

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
    def __init__(self, dim=256):
        super(LaplaceMask, self).__init__()
        #self.sinkmatch = Matching(dim)
        self.input_dim = dim

        self.lin1 = nn.Linear(dim, dim // 2, bias=False)
        self.lin2 = nn.Linear(dim // 2, dim // 4, bias=False)

        self.lin3 = nn.Linear(dim // 4, dim // 4, bias=False)
        self.lin4 = nn.Linear(dim // 4, dim // 8, bias=False)

        self.lin5 = nn.Linear(dim // 8, dim // 8, bias=False)
        self.lin6 = nn.Linear(dim // 8, 1, bias=False)

        self.attention = Spatial_Attention(dim // 4, num_heads=8, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_f, ref_f):
        #src_f B,N,C
        #ref_f B,M,C
        B,N,_ = src_f.shape
        _,M,_ = ref_f.shape
        x = torch.cat((src_f, ref_f), dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)

        x = self.attention(x)
        #(B,N+M,input_dim // 4)
        
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = self.relu(x)

        x = self.lin5(x)
        x = self.relu(x)
        x = self.lin6(x)

        src = x[:, :N, :]
        ref = x[:, N:N+M, :]
        x = src @ ref.transpose(1, 2)
        x = self.sigmoid(x)# B, N+M, 1
        
        return x.reshape(B, 1, N, M).contiguous()





