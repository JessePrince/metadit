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


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        mlp_bias: bool,
        qk_norm: bool,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} is not divisible by num heads {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp = SiGLU(dim, dim*3, mlp_bias)
        
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        
        self.pre_norm = nn.LayerNorm(dim)
        self.post_norm = nn. LayerNorm(dim)
        
    def forward(self, x: torch.Tensor):
        """
        x: B, N, D
        """
        B, N, C = x.shape
        
        # self-attention part
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
        
        # FFN part
        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
    
    
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
    
    
class Adaptor(nn.Module):
    def __init__(self, input_dim: int, tar_dim: tuple = (4, 32, 32), hidden_size=1024):
        """Text to image shape

        Args:
            input_dim (int): input dimension of the text feature
            tar_dim (int): target image dimension

        Raises:
            NotImplementedError: tar_dim must be 32768 = 8x64x64 or 16384 = 4x64x64
        """
        super().__init__()
        C, H, W = tar_dim
        assert H*W == hidden_size, f"Cannot convert {H*W} to {hidden_size}"
        self.tar_dim = C*H*W
        self.H = H
        self.W = W
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ln_fc1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln_fc2 = nn.LayerNorm(hidden_size)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.ln_conv1 = nn.LayerNorm([32, H, W])
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.ln_conv2 = nn.LayerNorm([64, H, W])
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=C, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor):
        """
        x: B, D
        """
        x = torch.relu(self.ln_fc1(self.fc1(x)))
        x = torch.relu(self.ln_fc2(self.fc2(x)))
        
        x = x.view(-1, 1, self.H, self.W)
        
        x = torch.relu(self.ln_conv1(self.conv1(x)))
        x = torch.relu(self.ln_conv2(self.conv2(x)))

        x = self.conv3(x)
        x = x.view(-1, self.tar_dim)
        
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        context_dim,
        dim,
        out_dim,
        num_heads,
        num_queries
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, dim)*0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(context_dim, dim*2)
        self.o_proj = nn.Linear(dim, out_dim)
        assert dim % num_heads == 0, f"dim {dim} not divisibile by heads {num_heads}"
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.num_queries = num_queries
        
    def forward(self, x: torch.Tensor):
        B, N, D = x.shape
        q = self.query.unsqueeze(0).expand(B, *self.query.shape)
        # q = torch.randn(B, self.num_queries, self.dim, device=x.device, dtype=x.dtype)
        q = self.q_proj(q).reshape(B, q.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = self.kv_proj(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.
        )
        x = x.transpose(1, 2).reshape(B, self.num_queries, self.dim)
        x = self.o_proj(x)
        
        return x


class SpectrumEncoder(nn.Module):
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
        use_patchify=False,
        patch_size=5,
        target_size=(4, 32, 32)
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} is not divisible by 2"
        length = None
        if use_patchify:
            self.spec_embedding = nn.Conv1d(
                in_channels=in_dim, 
                out_channels=dim, 
                kernel_size=patch_size, 
                stride=patch_size
            )
            length = math.floor((num_freqs - patch_size) / patch_size + 1)
        else:
            self.spec_embedding = nn.Linear(in_dim, dim)
            
        self.num_token = num_freqs if not use_patchify else length
        self.position_embedding = torch.tensor(get_1d_sincos_pos_embed_from_grid(dim, np.arange(self.num_token)), dtype=torch.float32)
        self.use_patchify = use_patchify
            
        self.blocks = nn.ModuleList()
        for idx, num in enumerate(num_blocks):
            for _ in range(num):
                self.blocks.append(TransformerDecoder(dim, num_heads, qkv_bias, mlp_bias, qk_norm))
                
            if idx != len(num_blocks)-1:
                self.blocks.append(nn.Linear(dim, dim // 2))
                dim = dim // 2
            
        self.final_layer = Adaptor(dim*self.num_token, target_size, 1024)
        self.target_size = target_size
        
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
        if not self.use_patchify:
            x = x.permute(0, 2, 1)
            x = self.spec_embedding(x)
        else:
            x = self.spec_embedding(x)
            x = x.permute(0, 2, 1)
        
        x = x + self.position_embedding.type_as(x)    
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_layer(x.view(x.shape[0],-1))
        
        return x
    
    
    
class SpectrumEncoder(nn.Module):
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
        use_patchify=False,
        patch_size=5,
        target_size=(4, 32, 32)
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} is not divisible by 2"
        length = None
        if use_patchify:
            self.spec_embedding = nn.Conv1d(
                in_channels=in_dim, 
                out_channels=dim, 
                kernel_size=patch_size, 
                stride=patch_size
            )
            length = math.floor((num_freqs - patch_size) / patch_size + 1)
        else:
            self.spec_embedding = nn.Linear(in_dim, dim)
            
        self.num_token = num_freqs if not use_patchify else length
        self.position_embedding = torch.tensor(get_1d_sincos_pos_embed_from_grid(dim, np.arange(self.num_token)), dtype=torch.float32)
        self.use_patchify = use_patchify
            
        self.blocks = nn.ModuleList()
        for idx, num in enumerate(num_blocks):
            for _ in range(num):
                self.blocks.append(TransformerDecoder(dim, num_heads, qkv_bias, mlp_bias, qk_norm))
                
            if idx != len(num_blocks)-1:
                self.blocks.append(nn.Linear(dim, dim // 2))
                dim = dim // 2
            
        self.final_layer = Adaptor(dim*self.num_token, target_size, 1024)
        self.target_size = target_size
        
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
        if not self.use_patchify:
            x = x.permute(0, 2, 1)
            x = self.spec_embedding(x)
        else:
            x = self.spec_embedding(x)
            x = x.permute(0, 2, 1)
        
        x = x + self.position_embedding.type_as(x)    
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_layer(x.view(x.shape[0],-1))
        
        return x
    
    
class SpectrumEncoderV2(nn.Module):
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
            
        self.num_token = num_freqs
        self.position_embedding = torch.tensor(get_1d_sincos_pos_embed_from_grid(dim, np.arange(self.num_token)), dtype=torch.float32)
            
        self.blocks = nn.ModuleList()
        for idx, num in enumerate(num_blocks):
            for _ in range(num):
                self.blocks.append(DualTransformerDecoder(dim, num_heads, num_freqs, qkv_bias, mlp_bias, qk_norm))
                
            # if idx != len(num_blocks)-1:
            #     self.blocks.append(nn.Linear(dim, dim // 2))
            #     dim = dim // 2
                
        self.proj_out = nn.Linear(dim, vae_channel)
        
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
        
        x = self.proj_out(x)
        
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
    
    
class SpectrumEncoderV3(nn.Module):
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
        vae_channel=8,
        latent_shape=32
    ):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} is not divisible by 2"
        self.spec_embedding = nn.Linear(in_dim, dim)
            
        self.num_token = num_freqs
        self.vae_channel = vae_channel
        self.latent_shape = latent_shape
        self.position_embedding = torch.tensor(get_1d_sincos_pos_embed_from_grid(dim, np.arange(self.num_token)), dtype=torch.float32)
        self.dim=dim
        self.blocks = nn.ModuleList()
        for idx, num in enumerate(num_blocks):
            for _ in range(num):
                self.blocks.append(DualTransformerDecoder(dim, num_heads, num_freqs, qkv_bias, mlp_bias, qk_norm))
                
            # if idx != len(num_blocks)-1:
            #     self.blocks.append(nn.Linear(dim, dim // 2))
            #     dim = dim // 2
                
        self.out_modulator = CrossAttention(dim, dim, dim, num_heads, latent_shape**2)
        self.quant_conv = nn.Conv2d(dim, vae_channel, kernel_size=7, padding=3)
        
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
        B, N, D = x.shape
        x = x.permute(0, 2, 1)
        x = self.spec_embedding(x)
        
        
        x = x + self.position_embedding.type_as(x)    
        
        for block in self.blocks:
            x = block(x)
        
        x = self.out_modulator(x)
        x = self.quant_conv(x.permute(0, 2, 1).reshape(B, self.dim, self.latent_shape, self.latent_shape))
        
        return x
    
class LinearModulator(nn.Module):
    def __init__(
        self,
        vae_channel,
        latent_shape,
        num_freqs,
        dim  
    ):
        super().__init__()
        self.proj1 = nn.Linear(vae_channel*latent_shape**2, vae_channel*num_freqs, bias=False)
        self.proj2 = nn.Linear(vae_channel, dim, bias=False)
        self.num_freqs = num_freqs
        self.dim = dim
        self.vae_channel = vae_channel
        
    def forward(self, x: torch.Tensor):
        """
        x: B, C, H, W
        """
        B = x.shape[0]
        x = self.proj1(x.reshape(B, -1))
        x = x.reshape(B, self.vae_channel, self.num_freqs).permute(0, 2, 1)
        return self.proj2(x)
    
class LazyLinearModulator(nn.Module):
    def __init__(
        self,
        vae_channel,
        latent_shape,
        num_freqs,
        dim  
    ):
        super().__init__()
        self.proj1 = nn.Linear(vae_channel*latent_shape**2, dim*num_freqs, bias=False)
        self.num_freqs = num_freqs
        self.dim = dim
        self.vae_channel = vae_channel
        
    def forward(self, x: torch.Tensor):
        """
        x: B, C, H, W
        """
        B = x.shape[0]
        x = self.proj1(x.reshape(B, -1))
        x = x.reshape(B, self.dim, self.num_freqs).permute(0, 2, 1)
        return x
        
    
class SpectrumDecoderV2(nn.Module):
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
        use_patchify=False,
        patch_size=5,
        vae_channel=8,
        latent_shape=32,
        use_cross_attn=True
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.vae_channel = vae_channel
        if use_cross_attn:
            self.in_modulator = CrossAttention(vae_channel, dim, dim, num_heads, num_freqs)
        else:
            self.in_modulator = LazyLinearModulator(vae_channel, latent_shape, num_freqs, dim)
            
        self.num_blocks = sum(num_blocks)
        self.num_token = num_freqs if not use_patchify else math.floor((num_freqs - patch_size) / patch_size + 1)
        
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(DualTransformerDecoder(dim, num_heads, self.num_token, qkv_bias, mlp_bias, qk_norm))
        
        self.final_proj = nn.Linear(dim, 2)
        
        self.apply(self._init_param)
        
    def _init_param(self, module):
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            
    def forward(self, x):
        if self.use_cross_attn:
            # B, vae_channel, H, W
            x = x.flatten(2).permute(0, 2, 1)
            x = self.in_modulator(x)
        else:
            x = self.in_modulator(x)
        for block in self.blocks:
            x = block(x)
            
        x = self.final_proj(x)
        
        return x.permute(0, 2, 1)
    
    
class SpectrumDecoder(nn.Module):
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
        use_patchify=False,
        patch_size=5,
        vae_channel=8
    ):
        super().__init__()
        self.proj_in = nn.Linear(vae_channel, dim)
        self.num_blocks = sum(num_blocks)
        self.num_token = num_freqs if not use_patchify else math.floor((num_freqs - patch_size) / patch_size + 1)
        
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(DualTransformerDecoder(dim, num_heads, self.num_token, qkv_bias, mlp_bias, qk_norm))
        
        self.final_proj = nn.Linear(dim, 2)
        
        self.apply(self._init_param)
        
    def _init_param(self, module):
        if hasattr(module, "bias"):
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            
    def forward(self, x):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
            
        x = self.final_proj(x)
        
        return x.permute(0, 2, 1)
    
    
    
if __name__ == "__main__":
    model = SpectrumEncoder(qkv_bias=True, mlp_bias=True, use_patchify=True)
    print(model)
    params = sum([p.nelement() for p in model.parameters()])
    print(params)
    x = torch.randn(2, 2, 301)
    output = model(x)
    print(output.shape)
    