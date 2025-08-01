"""
Surrogate model for S-param prediction, adopted from StarNet
https://github.com/ma-xu/Rewrite-the-Stars

"""
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from model.modeling_output import SurrogateOutput
from model.modules import SiGLU


class ConvBN(torch.nn.Sequential):
    def __init__(
        self, 
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        with_bn=True
    ):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class ConvSurrogate(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_freq=301*2, **kwargs):
        super().__init__()
        self.num_freq = num_freq
        assert self.num_freq % 2 == 0
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            SiGLU(self.in_channel, self.in_channel*2, mlp_bias=True),
            nn.Linear(self.in_channel, self.in_channel*2),
            nn.ReLU6(),
            nn.Linear(self.in_channel*2, num_freq)
        )
        # self.head = nn.Linear(self.in_channel, num_freq)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None):
        x = self.stem(inputs)
        for stage in self.stages:
            x = stage(x)
        # print(x.shape)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        x = self.head(x).reshape(x.shape[0], 2, self.num_freq // 2)
        
        loss = None
        if labels is not None:
            loss = F.l1_loss(x, labels)
        
        return SurrogateOutput(loss=loss, prediction=x)


def surrogate_s1(**kwargs):
    model = ConvSurrogate(24, [2, 2, 8, 3], **kwargs)
    
    return model


def surrogate_s2(**kwargs):
    model = ConvSurrogate(32, [1, 2, 6, 2], **kwargs)
    return model


def surrogate_s3(**kwargs):
    model = ConvSurrogate(32, [2, 2, 8, 4], **kwargs)
    return model


def surrogate_s4(**kwargs):
    model = ConvSurrogate(32, [3, 3, 12, 5], **kwargs)
    return model


# very small networks #
def surrogate_s050(**kwargs):
    return ConvSurrogate(16, [1, 1, 3, 1], 3, **kwargs)


def surrogate_s100(**kwargs):
    return ConvSurrogate(20, [1, 2, 4, 1], 4, **kwargs)


def surrogate_s150(**kwargs):
    return ConvSurrogate(24, [1, 2, 4, 2], 3, **kwargs)


if __name__ == "__main__":
    model = surrogate_s150()
    print(model)
    img = torch.randn(2, 3, 64, 64)
    lb = torch.randn(2, 2, 301)
    out = model(img, lb)
    print(out.loss)
    print(out.prediction.shape)
    