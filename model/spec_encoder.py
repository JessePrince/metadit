"""
Variational Encoder for spectrums
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import SiGLU
import math
import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
    
    
class DualTransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tokens: int,
        qkv_bias: bool,
        mlp_bias: bool,
        qk_norm: bool,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} is not divisible by num heads {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.qkv_proj2 = nn.Linear(num_tokens, num_tokens*3, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp = SiGLU(dim, dim*3, mlp_bias)
        
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        
        self.q_norm2 = nn.LayerNorm(self.num_tokens) if qk_norm else nn.Identity()
        self.k_norm2 = nn.LayerNorm(self.num_tokens) if qk_norm else nn.Identity()
        
        self.pre_norm = nn.LayerNorm(dim)
        self.mid_norm = nn.LayerNorm(self.num_tokens)
        self.post_norm = nn. LayerNorm(dim)
        
    def forward(self, x: torch.Tensor):
        """
        x: B, N, D
        """
        B, N, C = x.shape
        
        # temporal-attention part
        residual = x
        x = self.pre_norm(x)
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        x = residual + x
        
        # single head channel-attention part
        residual = x
        x = x.permute(0, 2, 1) # B, D, N
        x = self.mid_norm(x)
        qkv = self.qkv_proj2(x).reshape(B, C, 3, N).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm2(q), self.k_norm2(k)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.
        )
        x = x.permute(0, 2, 1) # B, N, D
        x = residual + x
        
        # FFN part
        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
class VanillaSpectrumEncoder(nn.Module):
    def __init__(
        self,
        num_blocks = [1, 1, 1, 1],
        in_dim=2,
        dim=256,
        num_freqs=301,
        num_heads=1,
        qkv_bias=True,
        mlp_bias=True,
        qk_norm=True,
        vae_channel=8
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} is not divisible by 2"
        self.spec_embedding = nn.Linear(in_dim, dim)
        self.dim = dim
            
        self.num_token = num_freqs
        self.position_embedding = torch.tensor(get_1d_sincos_pos_embed_from_grid(dim, np.arange(self.num_token)), dtype=torch.float32)
            
        self.blocks = nn.ModuleList()
        for idx, num in enumerate(num_blocks):
            for _ in range(num):
                self.blocks.append(DualTransformerDecoder(dim, num_heads, num_freqs, qkv_bias, mlp_bias, qk_norm))
                
            # if idx != len(num_blocks)-1:
            #     self.blocks.append(nn.Linear(dim, dim // 2))
            #     dim = dim // 2
                
        # self.proj_out = nn.Linear(dim, vae_channel)
        
        self.apply(self._init_param)
        
    def _init_param(self, module):
        if hasattr(module, "bias"):
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            
        
    def forward(self, x: torch.Tensor):
        """
        x: B, D, N
        """
        x = x.permute(0, 2, 1)
        x = self.spec_embedding(x)
        
        
        x = x + self.position_embedding.type_as(x)    
        
        for block in self.blocks:
            x = block(x)
        
        # x = self.proj_out(x)
        
        return x
