# @Time    : 2024/3/11 11:29
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from .transformer import LocalFeatureTransformer, LoFTREncoderLayer, PositionEncodingSine
from .correlation import AGCL


class Aggregation(nn.Module):
    def __init__(self, input_channel=[80, 160, 320, 640], group_wise_split=[4,4,4,4], search_num=[9,9,25,25], corr_split_mode = [1,1,0,0], downsample_scale=[4, 8, 16, 32], max_disp=192):
        super(Aggregation, self).__init__()

        self.dim = 16
        self.input_channel = input_channel

        self.self_att_fn = nn.ModuleList([LocalFeatureTransformer(d_model=input_channel[0], nhead=8, layer_names=['self']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=input_channel[1], nhead=8, layer_names=['self']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=input_channel[2], nhead=8, layer_names=['self']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=input_channel[3], nhead=8, layer_names=['self']*1, attention='linear')])
        self.cross_att_fn = nn.ModuleList([LocalFeatureTransformer(d_model=input_channel[0], nhead=8, layer_names=['cross']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=input_channel[1], nhead=8, layer_names=['cross']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=input_channel[2], nhead=8, layer_names=['cross']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=input_channel[3], nhead=8, layer_names=['cross']*1, attention='linear')])


        self.max_disp = max_disp
        self.group_wise_split = group_wise_split
        self.search_num = search_num
        self.corr_split_mode = corr_split_mode
        self.downsample_scale = downsample_scale

        self.att_encs = nn.ModuleList([LoFTREncoderLayer(d_model=self.search_num[0]*self.group_wise_split[0], nhead=4),
                            LoFTREncoderLayer(d_model=self.search_num[1]*self.group_wise_split[1], nhead=4),
                            LoFTREncoderLayer(d_model=self.search_num[2]*self.group_wise_split[2], nhead=4),
                            LoFTREncoderLayer(d_model=self.search_num[3]*self.group_wise_split[3], nhead=4)])
        
        self.conv_4 = nn.Conv2d(1, self.dim, 1)
        self.conv_4_out = nn.Conv2d(self.dim, 1, 1)
        self.upconv_4 = nn.ConvTranspose2d(self.dim, self.dim, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv_3 = nn.Conv2d(1, self.dim, 1)
        self.conv_34_out = nn.Conv2d(2*self.dim, out_channels=1, kernel_size=1)
        self.upconv_3 = nn.ConvTranspose2d(2*self.dim, self.dim, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_2 = nn.Conv2d(1, self.dim, 1)
        self.conv_23_out = nn.Conv2d(2*self.dim, out_channels=1, kernel_size=1)
        self.upconv_2 = nn.ConvTranspose2d(2*self.dim, self.dim, kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_1 = nn.Conv2d(1, self.dim, 1)
        self.conv_12_out = nn.Conv2d(2*self.dim, out_channels=1, kernel_size=1)
        self.upconv_1 = nn.ConvTranspose2d(2*self.dim, self.dim, kernel_size=4, stride=4, padding=0, bias=False)

        self.final_disp = nn.Conv2d(self.dim, 1, 1)

        self.activation = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
            """Custom weight initialization to ensure positivity."""
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    torch.nn.init.uniform_(m.weight, 0.00, 0.1)  # Positive values between 0.01 and 1.0
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.00)  # Ensure positive bias

    def forward(self, features_left, features_right):
        out_corr_stages = []
        out_disp = []
        out = []
        for i in range(4):
            N, C, H, W = features_left[i].size()

            position_encoding = PositionEncodingSine(d_model=self.input_channel[i], max_shape=(features_left[i].size(2), features_left[i].size(3)))
            x_tmp_left = position_encoding(features_left[i])
            x_tmp_right = position_encoding(features_right[i])

            features_left[i] = torch.reshape(
                torch.permute(x_tmp_left, (0, 2, 3, 1)), #(N, H, W, C)
                (N, H*W, C),
            )

            features_right[i] = torch.reshape(
                torch.permute(x_tmp_right, (0, 2, 3, 1)), #(N, H, W, C)
                (N, H*W, C),
            )

            features_left[i], features_right[i] = self.self_att_fn[i](features_left[i], features_right[i])
            features_left[i], features_right[i] = self.cross_att_fn[i](features_left[i], features_right[i])

            features_left[i], features_right[i] = [
                torch.permute(torch.reshape(x, (N, H, W, C)), (0,3,1,2)) #(N, C, H, W)
                for x in [features_left[i], features_right[i]]
            ]

            corr_method = AGCL(features_left[i], features_right[i], self.corr_split_mode[i])

            corr_res, disp_lists = corr_method(self.group_wise_split[i], self.search_num[i])# corr (N, search_num*split_num, H, W)
            out_corr_stages.append(corr_res)
            disp_lists = disp_lists.cuda()

            corr_seq = torch.reshape(
                torch.permute(corr_res, (0,2,3,1)), #(N, H, W, search_num*split_num)
                (N, H*W, self.search_num[i]*self.group_wise_split[i]),
            )

            cost_seq = self.att_encs[i](corr_seq, corr_seq, source_mask=None)

            attn_weights = F.softmax(cost_seq, dim=-1)
            disp_lists = disp_lists*self.downsample_scale[i]
            disp = torch.sum(attn_weights * disp_lists, axis=-1)
            disp = disp.reshape(N, 1, H, W)

            out_disp.append(disp)

        disp_4 = self.conv_4(out_disp[3]) #N, 16, H/32, W/32
        disp_4 = self.activation(disp_4)
        disp_4_out = self.conv_4_out(disp_4) #N, 1, H/32, W/32
        disp_4_out = self.activation(disp_4_out)
        out.append(disp_4_out)

        disp_4_up = self.upconv_4(disp_4) #N, 16, H/16, W/16
        disp_3 = self.conv_3(out_disp[2]) #N, 16, H/16, W/16

        disp_34 = torch.cat([disp_3, disp_4_up], dim=1) #N, 32, H/16, W/16
        disp_34 = self.activation(disp_34)
        disp_34_out = self.conv_34_out(disp_34) #N, 1, H/16, W/16
        disp_34_out = self.activation(disp_34_out)
        out.append(disp_34_out)

        disp_34_up = self.upconv_3(disp_34) #N, 16, H/8, W/8
        disp_2 = self.conv_2(out_disp[1]) #N, 16, H/8, W/8
        
        disp_234 = torch.cat([disp_2, disp_34_up], dim=1) #N, 32, H/8, W/8
        disp_234 = self.activation(disp_234)
        disp_234_out = self.conv_23_out(disp_234) #N, 1, H/8, W/8
        disp_234_out = self.activation(disp_234_out)
        out.append(disp_234_out)

        disp_234_up = self.upconv_2(disp_234) #N, 16, H/4, W/4
        disp_1 = self.conv_1(out_disp[0]) #N, 16, H/4, W/4

        disp_1234 = torch.cat([disp_1, disp_234_up], dim=1) #N, 32, H/4, W/4
        disp_1234 = self.activation(disp_1234)
        disp_1234_out = self.conv_12_out(disp_1234) #N, 1, H/4, W/4
        disp_1234_out = self.activation(disp_1234_out)
        out.append(disp_1234_out)

        disp_1234_up = self.upconv_1(disp_1234) #N, 16, H, W
        final_disp = self.final_disp(disp_1234_up) #N, 1, H, W
        final_disp = self.activation(final_disp)
        out.append(final_disp)

        return out
    

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