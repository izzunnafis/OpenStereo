# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume

import sys
import os

import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from .transformer import LocalFeatureTransformer, LoFTREncoderLayer, PositionEncodingSine
from .correlation import AGCL
from .mamba import VisionFoundationLayerDisp, VisionFoundationLayer

class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class FPNLayer(nn.Module):
    def __init__(self, chan_low, chan_high):
        super().__init__()
        self.deconv = BasicDeconv2d(chan_low, chan_high, kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.conv = BasicConv2d(chan_high * 2, chan_high, kernel_size=3, padding=1,
                                norm_layer=nn.BatchNorm2d,
                                act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

    def forward(self, low, high):
        low = self.deconv(low)
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


class Aggregation(nn.Module):
    def __init__(self, input_channel=[24, 32, 96], group_wise_split_num=[4,4,4], search_num=[49,36,25], corr_split_mode = [1,1,1], downsample_scale=[4, 8, 16], max_disp=192):
        super(Aggregation, self).__init__()

        self.attention_channels = [48, 64, 96]

        self.input_channel = input_channel
        self.max_disp = max_disp
        self.group_wise_split_num = group_wise_split_num
        self.search_num = search_num
        self.corr_split_mode = corr_split_mode
        self.downsample_scale = downsample_scale

        self.corr_disp = [self.group_wise_split_num[0]*self.search_num[0],
                          self.group_wise_split_num[1]*self.search_num[1],
                          self.group_wise_split_num[2]*self.search_num[2]]

        self.conv0_init = MobileV2Residual(self.corr_disp[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        self.conv1 = MobileV2Residual(self.attention_channels[0], self.attention_channels[1], stride=2, expanse_ratio=4)
        self.conv2 = MobileV2Residual(self.attention_channels[1], self.attention_channels[2], stride=2, expanse_ratio=4)

        # conv0 = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        #          for i in range(3)]
        # self.conv0 = nn.Sequential(*conv0)

        # self.conv1_init = MobileV2Residual(self.corr_disp[1], self.attention_channels[1], stride=1, expanse_ratio=4)
        # conv1 = [MobileV2Residual(self.attention_channels[1], self.attention_channels[1], stride=1, expanse_ratio=4)
        #             for i in range(3)]
        # self.conv1 = nn.Sequential(*conv1)

        # self.conv2_init = MobileV2Residual(self.corr_disp[2], self.attention_channels[2], stride=1, expanse_ratio=4)
        # conv2 = [MobileV2Residual(self.attention_channels[2], self.attention_channels[2], stride=1, expanse_ratio=4)
        #             for i in range(3)]
        # self.conv2 = nn.Sequential(*conv2)

        # self.conv_no0 = BasicConv2d(self.input_channel[0], self.attention_channels[0], kernel_size=3, padding=1,
        #                         norm_layer=nn.BatchNorm2d,
        #                         act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
        # self.conv_up21 = BasicDeconv2d(self.input_channel[2], self.attention_channels[1], kernel_size=4, stride=2, padding=1,
        #                             norm_layer=nn.BatchNorm2d,
        #                             act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))
        # self.conv_up22 = BasicDeconv2d(self.attention_channels[1], self.attention_channels[0], kernel_size=4, stride=2, padding=1,
        #                             norm_layer=nn.BatchNorm2d,
        #                             act_layer=partial(nn.LeakyReLU, negative_slope=0.2, inplace=True))

        self.attention_feat = AttentionModule(self.attention_channels[0], self.input_channel[0])

        # self.feat_l_init = MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        # feat_l = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        #                              for i in range(2)]
        # self.feat_l = nn.Sequential(*feat_l)
        
        # self.feat_h_init = MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        # self.feat_h = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        #                                 for i in range(2)]
        # self.feat_h = nn.Sequential(*self.feat_h)


        # self.fpn_layer0 = FPNLayer(self.attention_channels[0], self.attention_channels[0])
        self.fpn_layer1 = FPNLayer(self.attention_channels[1], self.attention_channels[0])
        self.fpn_layer2 = FPNLayer(self.attention_channels[2], self.attention_channels[1])   

        conv1_0 = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
                    for i in range(2)]
        self.conv1_0 = nn.Sequential(*conv1_0)

        conv1_1 = [MobileV2Residual(self.attention_channels[1], self.attention_channels[1], stride=1, expanse_ratio=4)
                       for i in range(2)]
        self.conv1_1 = nn.Sequential(*conv1_1)

        conv2_0 = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
                    for i in range(2)]
        self.conv2_0 = nn.Sequential(*conv2_0)


        self.conv_l_down0 = MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=2, expanse_ratio=4)
        self.conv_l_down1 = MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=2, expanse_ratio=4)
        self.conv_l_up1 =  BasicDeconv2d(self.attention_channels[0], self.attention_channels[0], kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.ReLU6, inplace=True))
        self.conv_l_up0 = BasicDeconv2d(self.attention_channels[0], self.attention_channels[0], kernel_size=4, stride=2, padding=1,
                                    norm_layer=nn.BatchNorm2d,
                                    act_layer=partial(nn.ReLU6, inplace=True))
        # conv_l = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
        #             for i in range(2)]
        # self.conv_l = nn.Sequential(*conv_l)

        conv_h = [MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=4)
                 for i in range(4)]
        self.conv_h = nn.Sequential(*conv_h)


        self.redir0 = MobileV2Residual(self.attention_channels[0], self.attention_channels[0], stride=1, expanse_ratio=1/4)
        self.redir1 = MobileV2Residual(self.attention_channels[1], self.attention_channels[1], stride=1, expanse_ratio=1/4)

    def _init_weights(self):
            """Custom weight initialization to ensure positivity."""
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    torch.nn.init.uniform_(m.weight, 0.00, 0.1)  # Positive values between 0.01 and 1.0
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.00)  # Ensure positive bias

    def process_corr(self, features_left, features_right, left_img, right_img, stage_idx):

        # Compute correlation and cost volume
        corr_method = AGCL(features_left, features_right, self.corr_split_mode[stage_idx])
        corr_res, _ = corr_method(self.group_wise_split_num[stage_idx], self.search_num[stage_idx])
        
        return corr_res

    def forward(self, features_left, features_right, left_img, right_img):

        # start_time = time.time()
        corr0 = self.process_corr(features_left[0], features_right[0], left_img, right_img, 0)  # N, search_num*group_wise_split_num, H, W
        # torch.cuda.synchronize()
        # print(f"Time for process_corr: {time.time() - start_time:.4f}s")

        # start_time = time.time()
        vol0 = self.conv0_init(corr0)  # N, attention_channels[0], H, W
        vol1 = self.conv1(vol0)  # N, attention_channels[0], H, W
        vol2 = self.conv2(vol1)  # N, attention_channels[0], H, W
        # torch.cuda.synchronize()
        # print(f"Time for volume processing: {time.time() - start_time:.4f}s")

        # start_time = time.time()
        feat = self.attention_feat(features_left[0])  # N, attention_channels[0], H, W
        # torch.cuda.synchronize()
        # print(f"Time for attention feature processing: {time.time() - start_time:.4f}s")

        # start_time = time.time()
        feat_l = F.sigmoid(feat)
        feat_h = 1 - feat_l
        # torch.cuda.synchronize()
        # print(f"Time for feature splitting: {time.time() - start_time:.4f}s")

        # start_time = time.time()
        vol_att12 = self.fpn_layer2(vol2, vol1)  # N, attention_channels[1], H, W
        vol_att12 = self.conv1_1(vol_att12) + self.redir1(vol1)  # N, attention_channels[1], H, W
        vol_att0 = self.conv1_0(vol0)  # N, attention_channels[0], H, W
        vol_att012 = self.fpn_layer1(vol_att12, vol_att0)  # N, attention_channels[0], H, W
        vol_att012 = self.conv2_0(vol_att012) + self.redir0(vol0)  # N, attention_channels[0], H, W
        # torch.cuda.synchronize()
        # print(f"Time for FPN and redirection layers: {time.time() - start_time:.4f}s")

        # start_time = time.time()
        vol_l = vol_att012 * feat_l  # N, attention_channels[0], H, W
        vol_h = vol_att012 * feat_h
        vol_l_1_down = self.conv_l_down0(vol_l)  # N, attention_channels[0], H/2, W/2
        vol_l_2_down = self.conv_l_down1(vol_l_1_down)  # N, attention_channels[0], H/4, W/4
        vol_l_1_up = self.conv_l_up1(vol_l_2_down) + vol_l_1_down  # N, attention_channels[0], H/2, W/2
        vol_l = self.conv_l_up0(vol_l_1_up) + vol_l  # N, attention_channels[0], H, W
        vol_h = self.conv_h(vol_h)  # N, attention_channels[0], H, W
        vol_all = F.relu(vol_l * feat_l + vol_h * feat_h, inplace=True)  # N, attention_channels[0], H, W
        # torch.cuda.synchronize()
        # print(f"Time for final volume computation: {time.time() - start_time:.4f}s")
        return vol_all, feat_l, feat_h
    
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
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        # v2
        feat = self.pwconv(x)
        feat = self.dwconv(feat)
        feat = self.pwliner(feat)

        if self.use_res_connect:
            return x + feat
        else:
            return feat

class MobileOneBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.identity = nn.BatchNorm2d(in_ch) if stride == 1 and in_ch == out_ch else None
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_3x3(x) + self.conv_1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return self.act(out)
    

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

    def forward(self, x):
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn


if __name__ == "__main__":
    import torch.optim as optim

    model = Aggregation().cuda()
    model.train()
    print(model)

    # Create dummy input data
    features_left = [torch.randn(1, 256, 64, 64), torch.randn(1, 256, 32, 32), torch.randn(1, 256, 16, 16), torch.randn(1, 256, 8, 8)]
    features_right = [torch.randn(1, 256, 64, 64), torch.randn(1, 256, 32, 32), torch.randn(1, 256, 16, 16), torch.randn(1, 256, 8, 8)]

    # Convert features to CUDA
    features_left = [f.cuda() for f in features_left]
    features_right = [f.cuda() for f in features_right]

    # Forward pass
    out = model(features_left, features_right)
    for o in out:
        print(o.size())

    # Check output type and shape
    import unittest
    unittest.TestCase().assertIsInstance(out, list)
    unittest.TestCase().assertEqual(len(out), 5)
    unittest.TestCase().assertEqual(out[0].shape, torch.Size([1, 1, 8, 8]))
    unittest.TestCase().assertEqual(out[1].shape, torch.Size([1, 1, 16, 16]))
    unittest.TestCase().assertEqual(out[2].shape, torch.Size([1, 1, 32, 32]))
    unittest.TestCase().assertEqual(out[3].shape, torch.Size([1, 1, 64, 64]))
    unittest.TestCase().assertEqual(out[4].shape, torch.Size([1, 1, 256, 256]))
    

    # Forward pass
    output = model(features_left, features_right)

    # Create dummy target and loss
    target = torch.randn_like(output[-1])
    criterion = torch.nn.MSELoss()
    loss = criterion(output[-1], target)

    target2 = torch.rand_like(output[-2])
    loss += criterion(output[-2], target2)

    target3 = torch.rand_like(output[-3])
    loss += criterion(output[-3], target3)

    target4 = torch.rand_like(output[-4])
    loss += criterion(output[-4], target4)

    target5 = torch.rand_like(output[-5])
    loss += criterion(output[-5], target5)

    # Retain gradients for debugging
    for feature in features_left + features_right:
        feature.retain_grad()

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward(retain_graph=True)

    # Debug: Check if gradients exist
    for name, param in model.named_parameters():
        print(name)
        if param.grad is None:
            print(f"⚠️ Gradient is None for parameter: {name}")
        elif torch.all(param.grad == 0):
            print(f"⚠️ Gradient is all zeros for parameter: {name}")

    # Save initial weights
    initial_weights = [param.clone() for param in model.parameters()]

    # Update weights
    optimizer.step()

    # Check if weights are updated
    for initial, updated in zip(initial_weights, model.parameters()):
        assert not torch.equal(initial, updated), f"🚨 Weights not updated for {updated}"