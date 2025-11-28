import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F
import math

class SiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
        

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
