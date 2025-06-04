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

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(blocks[0])]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1 = MobileV2Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)
        conv2_add = [MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[1] - 1)]
        self.conv2 = nn.Sequential(*conv2_add)

        self.conv3 = MobileV2Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)
        conv4_add = [MobileV2Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(blocks[2] - 1)]
        self.conv4 = nn.Sequential(*conv4_add)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])
            self.att2 = AttentionModule(in_channels * 2, backbone_channels[1])
            self.att4 = AttentionModule(in_channels * 4, backbone_channels[2])

        self.conv_l_down0 = MobileV2Residual(in_channels, in_channels, stride=2, expanse_ratio=4)
        self.conv_l_down1 = MobileV2Residual(in_channels, in_channels, stride=2, expanse_ratio=4)
        self.conv_l_up1 =  BasicDeconv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.ReLU6, inplace=True))
        self.conv_l_up0 = BasicDeconv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.ReLU6, inplace=True))
        
        conv_h = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=4)
                 for i in range(4)]
        self.conv_h = nn.Sequential(*conv_h)


        

    def forward(self, x, features_left, freq_filter=None):
        x = self.conv0(x)
        if self.left_att:
            x = self.att0(x, features_left[0])

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.left_att:
            conv2 = self.att2(conv2, features_left[1])

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        if self.left_att:
            conv4 = self.att4(conv4, features_left[2])

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        if freq_filter is not None:
            feat_h = F.sigmoid(freq_filter)
            feat_l = 1 - feat_h
            conv6_h = conv6 * feat_h
            conv6_l = conv6 * feat_l

            conv7_h = self.conv_h(conv6_h)
            conv7_l_1_down = self.conv_l_down0(conv6_l)
            conv7_l_2_down = self.conv_l_down1(conv7_l_1_down)
            conv7_l_1_up = self.conv_l_up1(conv7_l_2_down) + conv7_l_1_down
            conv7_l = self.conv_l_up0(conv7_l_1_up) + conv6_l

            conv7_res = F.relu(conv7_h + conv7_l, inplace=True)
            # conv7_res = F.relu(conv7_h*feat_h + conv7_l*feat_l, inplace=True)
        
        else:
            conv7_res = conv6

        return [conv7_res]


class Aggregation2(nn.Module):
    def __init__(self, in_channels, left_att, blocks, expanse_ratio, backbone_channels):
        super(Aggregation2, self).__init__()

        self.left_att = left_att
        self.expanse_ratio = expanse_ratio

        conv0 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(4)]
        self.conv0 = nn.Sequential(*conv0)

        self.conv1_l = MobileV2Residual(in_channels, in_channels, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2_l = MobileV2Residual(in_channels, in_channels, stride=2, expanse_ratio=self.expanse_ratio)
        conv3 = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                     for i in range(4)]
        self.conv3_l = nn.Sequential(*conv3)

        self.conv4_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.conv5_l = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))
        
        conv1_h = [MobileV2Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
                 for i in range(4)]
        self.conv_h = nn.Sequential(*conv1_h)


        if self.left_att:
            self.att0 = AttentionModule(in_channels, backbone_channels[0])



        

    def forward(self, x, features_left, freq_filter=None):
        feat_h = F.sigmoid(freq_filter)
        feat_l = 1 - feat_h

        if self.left_att:
            x = self.att0(x, features_left[0])

        x = self.conv0(x)

        conv_h = x * feat_h
        conv_l = x * feat_l

        conv1_l = self.conv1_l(conv_l)
        conv2_l = self.conv2_l(conv1_l)

        conv3_l = self.conv3_l(conv2_l)
        conv4_l = self.conv4_l(conv3_l) + conv1_l
        conv5_l = self.conv5_l(conv4_l)

        conv_h = self.conv_h(conv_h)

        conv_6 = F.relu(conv_h*feat_h + conv5_l*feat_l, inplace=True)

        return [conv_6]


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
