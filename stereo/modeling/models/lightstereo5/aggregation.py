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

        self.left_att = True
        self.expanse_ratio = expanse_ratio
        
        self.blocks = [1, 1, 1]
    
        conv0 = [DisparityBoostResidual(in_channels)
                 for i in range(self.blocks[0])]
        self.conv0 = nn.Sequential(*conv0)
        
        conv1 = [DisparityBoostResidual(in_channels)
                 for i in range(self.blocks[1])]
        self.conv1 = nn.Sequential(*conv1)
        
        conv2 = [DisparityBoostResidual(in_channels)
                    for i in range(self.blocks[2])]
        self.conv2 = nn.Sequential(*conv2)

        conv0 = [MobileV2Residual(in_channels)
                 for i in range(self.blocks[0])]
        self.conv0 = nn.Sequential(*conv0)
        
        conv1 = [MobileV2Residual(in_channels)
                 for i in range(self.blocks[1])]
        self.conv1 = nn.Sequential(*conv1)
        
        conv2 = [MobileV2Residual(in_channels)
                    for i in range(self.blocks[2])]
        self.conv2 = nn.Sequential(*conv2)
        
        self.att1 = AttentionModule(in_channels, backbone_channels[0])
        self.att2 = AttentionModule(in_channels, backbone_channels[0])
                

    def forward(self, x, features_left=None, freq_filter=None, concat_vol=None):
        
        conv0 = self.conv0(x)
        conv0 = self.att1(conv0, features_left[0])
        conv1 = self.conv1(conv0)
        conv1 = self.att2(conv1, features_left[0])
        conv2 = self.conv2(conv1)
        
        return [conv2]


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
            # nn.Conv2d(hidden_dim, hidden_dim, (3, 1), 1, padding=(1,0), bias=False),
            # nn.Conv2d(hidden_dim, hidden_dim, (1, 3), 1, padding=(0,1), bias=False),
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

class DisparityBoostResidual(nn.Module):
    def __init__(self, inp, dilation=1):
        super(DisparityBoostResidual, self).__init__()
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True)

        )
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(inp, inp//2, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp//2),
            nn.ReLU6(inplace=True)
        )
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(inp, inp//4, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp//4),
            nn.ReLU6(inplace=True)
        )
        self.disp_conv = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=1, padding=1, bias=False))
        self.disp_conv1 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=1, padding=1, bias=False))
        self.disp_conv2 = nn.Sequential(
            nn.Conv3d(1, 1, 3, stride=1, padding=1, bias=False))
        
        self.pwliner = nn.Sequential(
            nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp)
        )
        self.pwliner1 = nn.Sequential(
            nn.Conv2d(inp//2, inp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp)
        )
        self.pwliner2 = nn.Sequential(
            nn.Conv2d(inp//4, inp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp)
        )

    def forward(self, inputs):
        feat = self.pwconv(inputs)
        feat1 = self.pwconv1(inputs)
        feat2 = self.pwconv2(inputs)
        feat = feat.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        feat = self.disp_conv(feat)  # [B, 1, C, H, W] -> [B, 1, 1, H, W]
        feat = feat.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
        feat1 = feat1.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        feat1 = self.disp_conv1(feat1)  # [B, 1, C, H, W] -> [B, 1, 1, H, W]
        feat1 = feat1.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
        feat2 = feat2.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        feat2 = self.disp_conv2(feat2)  # [B, 1, C, H, W] -> [B, 1, 1, H, W]
        feat2 = feat2.squeeze(1)  # [B, 1, C, H, W] -> [B, C, H, W]
        feat = self.pwliner(feat)
        feat1 = self.pwliner1(feat1)
        feat2 = self.pwliner2(feat2)
        feat = feat + feat1 + feat2
        return feat


class PointWiseResidual(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(PointWiseResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        # v2
        self.pwconv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
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
        self.dwconv = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        # self.sfa = c_att(hidden_dim, stride=stride, ks=7, groups=4, gamma=1.4, b=1.4)
        self.pwliner = nn.Sequential(
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup)
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


class MobileV2ResidualCA(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2ResidualCA, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        self.ca = CoordAtt(hidden_dim, hidden_dim)

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
        feat = self.ca(feat)
        # feat = self.sfa(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat

class MobileNextResidual(nn.Module):
    def __init__(self,inp, oup, stride, expanse_ratio, dilation=1):
        super().__init__()

        # Expansion phase
        self.inp = inp
        self.hidden_dim = int(inp // expanse_ratio)
        self.oup = oup
        self.res_connect = self.inp == self.oup and stride == 1
        k = 3
        s = stride

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.inp, out_channels=self.inp, kernel_size=k, bias=False, groups=self.inp, padding=1),
            nn.BatchNorm2d(num_features=self.inp),
            Swish(),
            #first linear layer
            nn.Conv2d(in_channels=self.inp, out_channels=self.hidden_dim, kernel_size=1, bias=False, groups=1),
            nn.BatchNorm2d(num_features=self.hidden_dim),
            # sec linear layer
            nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.oup, kernel_size=1, bias=False, groups=1),
            nn.BatchNorm2d(num_features=self.oup),
            Swish(),
            # expand layer
            nn.Conv2d(in_channels=self.oup, out_channels=self.oup, kernel_size=k, bias=False, groups = self.oup, stride=s, padding=1),
            nn.BatchNorm2d(num_features=self.oup),
            )


    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        x = self.features(inputs)

        # Skip connection and drop connect
        if self.res_connect:
            x = x + inputs
        return x

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
