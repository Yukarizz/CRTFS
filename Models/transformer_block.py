# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
from matplotlib import pyplot as plt
import matplotlib
from torch.nn.functional import interpolate
matplotlib.use("agg")
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class MutualAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.rgb_proj = nn.Linear(dim, dim)
#
#         self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.depth_proj = nn.Linear(dim, dim)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, rgb_fea, depth_fea):
#         B, N, C = rgb_fea.shape
#
#         rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         # q [B, nhead, N, C//nhead]
#
#         depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#         depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         # rgb branch
#         rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
#         rgb_attn = rgb_attn.softmax(dim=-1)
#         rgb_attn = self.attn_drop(rgb_attn)
#
#         rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
#         rgb_fea = self.rgb_proj(rgb_fea)
#         rgb_fea = self.proj_drop(rgb_fea)
#
#         # depth branch
#         depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
#         depth_attn = depth_attn.softmax(dim=-1)
#         depth_attn = self.attn_drop(depth_attn)
#
#         depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
#         depth_fea = self.depth_proj(depth_fea)
#         depth_fea = self.proj_drop(depth_fea)
#
#         return rgb_fea, depth_fea
class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.color_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(dim, dim)

        # self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.Second_reduce = nn.Linear(3*dim, dim)

    def visualize_average_attention_upsampled(self, attn_map, patch_size=14, head_rows=2, head_cols=3, scale_factor=4):
        """
        可视化放大后的平均多头注意力图

        参数:
            attn_map: 注意力图，形状为 [1, num_heads, num_patches, num_patches]
            patch_size: 原始图像被分成的patch大小 (14表示14x14)
            head_rows: 显示多头注意力时的行数
            head_cols: 显示多头注意力时的列数
            scale_factor: 插值放大的倍数
        """
        attn_map = attn_map.squeeze(0)  # [6, 196, 196]
        num_heads = attn_map.shape[0]

        # 准备画布
        fig, axes = plt.subplots(head_rows, head_cols, figsize=(head_cols * 4, head_rows * 4))

        for i, ax in enumerate(axes.flat):
            if i >= num_heads:
                ax.axis('off')
                continue

            # 计算平均注意力并reshape为2D
            avg_attn = attn_map[i].mean(dim=0).reshape(1, 1, patch_size, patch_size)

            # 转换为tensor并进行双三次插值放大
            avg_attn_tensor = torch.tensor(avg_attn, dtype=torch.float32)
            upsampled_attn = interpolate(avg_attn_tensor,
                                         scale_factor=scale_factor,
                                         mode='bicubic',
                                         align_corners=False)

            # 转换为numpy数组并移除额外维度
            upsampled_attn = upsampled_attn.squeeze().cpu().numpy()

            # 可视化
            im = ax.imshow(upsampled_attn, cmap='jet', interpolation='none')
            ax.set_title(f'Head {i + 1} (Upsampled {scale_factor}x)')
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=0.6)

        plt.suptitle(f'Average Attention (Bicubic Upsampling {scale_factor}x)', y=1.02)
        plt.tight_layout()
        plt.savefig("1.png")
    def forward(self, rgb_fea, depth_fea,color_fea):
        B, N, C = rgb_fea.shape

        color_q = self.color_q(color_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        # depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # rgb branch
        rgb_attn = (color_q @ rgb_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea_refine = (rgb_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea_refine = self.rgb_proj(rgb_fea_refine)
        rgb_fea_refine = self.proj_drop(rgb_fea_refine) + rgb_fea

        # depth branch
        rgb_q = self.rgb_q(rgb_fea_refine).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        depth_attn = (depth_attn).softmax(dim=-1)
        # self.visualize_average_attention_upsampled(depth_attn)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea_refine = (depth_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        depth_fea_refine = self.depth_proj(depth_fea_refine)
        depth_fea_refine = self.proj_drop(depth_fea_refine) + depth_fea

        fuse_fea = self.Second_reduce(torch.cat([self.gamma1*rgb_fea_refine,self.gamma2*depth_fea_refine,self.gamma3*color_fea],dim=2))
        return fuse_fea

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MutualSelfBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        # mutual attention
        self.norm1_rgb_ma = norm_layer(dim)
        self.norm2_depth_ma = norm_layer(dim)
        self.norm3_color_ma = norm_layer(dim)
        self.mutualAttn = MutualAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm3_rgb_ma = norm_layer(dim)
        self.norm4_depth_ma = norm_layer(dim)
        self.mlp_rgb_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_depth_ma = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # rgb self attention
        self.norm1_rgb_sa = norm_layer(dim)
        self.selfAttn_rgb = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_rgb_sa = norm_layer(dim)
        self.mlp_rgb_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # depth self attention
        self.norm1_depth_sa = norm_layer(dim)
        self.selfAttn_depth = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2_depth_sa = norm_layer(dim)
        self.mlp_depth_sa = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, rgb_fea, depth_fea, color_fea):

        # mutual attention
        fuse_fea = self.drop_path(self.mutualAttn(self.norm1_rgb_ma(rgb_fea), self.norm2_depth_ma(depth_fea),self.norm3_color_ma(color_fea)))

        rgb_fea = rgb_fea + fuse_fea
        depth_fea = depth_fea + fuse_fea

        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_ma(self.norm3_rgb_ma(rgb_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_ma(self.norm4_depth_ma(depth_fea)))

        # rgb self attention
        rgb_fea = rgb_fea + self.drop_path(self.selfAttn_rgb(self.norm1_rgb_sa(rgb_fea)))
        rgb_fea = rgb_fea + self.drop_path(self.mlp_rgb_sa(self.norm2_rgb_sa(rgb_fea)))

        # depth self attention
        depth_fea = depth_fea + self.drop_path(self.selfAttn_depth(self.norm1_depth_sa(depth_fea)))
        depth_fea = depth_fea + self.drop_path(self.mlp_depth_sa(self.norm2_depth_sa(depth_fea)))

        return rgb_fea, depth_fea


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
