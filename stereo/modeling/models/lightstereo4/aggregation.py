# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
import math
import torch
from .submodule import CoordAtt, Swish


class Aggregation(nn.Module):
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation, self).__init__()

        self.expanse_ratio = 4
        
        self.blocks = [4, 4, 2]

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        self.conv2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        # self.conv2_dup = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        
        self.conv1_3d = nn.Sequential(*[
            MobileV2Residual3D(8, 8, stride=1, expanse_ratio=self.expanse_ratio)
            for i in range(self.blocks[0])
        ])
        
        self.conv3_comb_rep = nn.Sequential(*[
            MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
            for i in range(self.blocks[1])
        ])

        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv_l = nn.Sequential(*[
            MobileV2Residual(4*in_channels, 4*in_channels, stride=1, expanse_ratio=self.expanse_ratio)
            for i in range(self.blocks[2])
        ])

        self.conv_l_up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv_l_up0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))
        
        conv_h = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(self.blocks[2])]
        self.conv_h = nn.Sequential(*conv_h)


    def forward(self, x, vol_3d=None):
        conv1 = self.conv1(x)
            
        if vol_3d is not None:
            conv2 = self.conv2(conv1)
            B, C, H, W = conv2.shape
            conv2_3d = self.conv1_3d(vol_3d)
            conv2_3d = conv2_3d.view(B, C, H, W)
            conv3 = self.conv3_comb_rep(torch.cat([conv2, conv2_3d], dim=1))
        else:
            conv2 = self.conv2_dup(conv1)
            conv3 = self.conv3_comb_rep(conv2)

        conv_redir1 = self.redir1(x)
        # conv_redir2 = self.redir2(conv1)

        conv4 = F.relu(self.conv4(conv3), inplace=True)
        conv5 = F.relu(self.conv5(conv4) + conv_redir1, inplace=True)

        conv6_h = self.conv_h(conv5)
        conv6_l = self.conv_l(conv3)

        conv6_l_1_up = self.conv_l_up1(conv6_l)
        conv7_l = self.conv_l_up0(conv6_l_1_up) + conv_redir1

        conv7_res = F.relu(conv6_h + conv7_l, inplace=True)
        
        return [conv7_res]


class MobileV2Residual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        # self.sfa = c_att(hidden_dim, stride=stride, ks=7, groups=4, gamma=1.4, b=1.4)
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        # feat = self.sfa(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat

class MobileV2Residual3D(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2Residual3D, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dwconv331 = nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, dilation=dilation, groups=hidden_dim, bias=False)
        # self.dwconv332 = nn.Conv3d(hidden_dim, hidden_dim, 5, stride, 2, dilation=dilation, groups=hidden_dim, bias=False)
        # self.dwconv31 = nn.Conv3d(hidden_dim, hidden_dim, [3, 1, 1], stride, [1, 0, 0], dilation=dilation, groups=hidden_dim, bias=False)
        # self.dwconv32 = nn.Conv3d(hidden_dim, hidden_dim, [5, 1, 1], stride, [2, 0, 0], dilation=dilation, groups=hidden_dim, bias=False)
        # self.dwconv33 = nn.Conv3d(hidden_dim, hidden_dim, [7, 1, 1], stride, [3, 0, 0], dilation=dilation, groups=hidden_dim, bias=False)
        # self.dwconv34 = nn.Conv3d(hidden_dim, hidden_dim, [11, 1, 1], stride, [5, 0, 0], dilation=dilation, groups=hidden_dim, bias=False)
        self.dwconv = nn.Sequential(
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
        feat = self.dwconv331(feat)
        feat = self.dwconv(feat) 
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat

class AttentionModule(nn.Module):
    def __init__(self, dim, img_feat_dim):
        super().__init__()
        self.conv0 = nn.Conv2d(img_feat_dim, dim, 1)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, cost, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * cost
