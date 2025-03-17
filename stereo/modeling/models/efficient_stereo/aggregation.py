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

        self.attention_channels = [128, 128, 256, 256]

        self.dim = 16
        self.input_channel = input_channel

        self.self_conv_fn1 = nn.ModuleList([nn.Conv2d(input_channel[0], self.attention_channels[0], 3, stride=1, padding=1),
                                          nn.Conv2d(input_channel[1], self.attention_channels[1], 3, stride=1, padding=1),
                                          nn.Conv2d(input_channel[2], self.attention_channels[2], 3, stride=1, padding=1),
                                          nn.Conv2d(input_channel[3], self.attention_channels[3], 3, stride=1, padding=1)])

        self.cross_att_fn = nn.ModuleList([LocalFeatureTransformer(d_model=self.attention_channels[0], nhead=8, layer_names=['cross']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=self.attention_channels[1], nhead=8, layer_names=['cross']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=self.attention_channels[2], nhead=8, layer_names=['cross']*1, attention='linear'),
                            LocalFeatureTransformer(d_model=self.attention_channels[3], nhead=8, layer_names=['cross']*1, attention='linear')])

        self.self_conv_fn2 = nn.ModuleList([nn.Conv2d(self.attention_channels[0], self.attention_channels[0], 3, stride=1, padding=1),
                                          nn.Conv2d(self.attention_channels[1], self.attention_channels[0], 3, stride=1, padding=1),
                                          nn.Conv2d(self.attention_channels[2], self.attention_channels[0], 3, stride=1, padding=1),
                                          nn.Conv2d(self.attention_channels[3], self.attention_channels[0], 3, stride=1, padding=1)])


        self.max_disp = max_disp
        self.group_wise_split = group_wise_split
        self.search_num = search_num
        self.corr_split_mode = corr_split_mode
        self.downsample_scale = downsample_scale

        self.att_encs = nn.ModuleList([LoFTREncoderLayer(d_model=self.search_num[0]*self.group_wise_split[0], nhead=4),
                            LoFTREncoderLayer(d_model=self.search_num[1]*self.group_wise_split[1], nhead=4),
                            LoFTREncoderLayer(d_model=self.search_num[2]*self.group_wise_split[2], nhead=4),
                            LoFTREncoderLayer(d_model=self.search_num[3]*self.group_wise_split[3], nhead=4)])
        
        self.nn_linears = nn.ModuleList([nn.Linear(self.search_num[0]*self.group_wise_split[0], 17),
                                        nn.Linear(self.search_num[1]*self.group_wise_split[1], 17),
                                        nn.Linear(self.search_num[2]*self.group_wise_split[2], 17),
                                        nn.Linear(self.search_num[3]*self.group_wise_split[3], 17)])
        
        self.upconv_4 = nn.ConvTranspose2d(self.attention_channels[0], self.attention_channels[0], kernel_size=4, stride=2, padding=1, bias=False)
        
        self.conv_3 = nn.Conv2d(2*self.attention_channels[0], self.attention_channels[0], 1)
        self.upconv_3 = nn.ConvTranspose2d(self.attention_channels[0], self.attention_channels[0], kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_2 = nn.Conv2d(2*self.attention_channels[0], self.attention_channels[0], 1)
        self.upconv_2 = nn.ConvTranspose2d(self.attention_channels[0], self.attention_channels[0], kernel_size=4, stride=2, padding=1, bias=False)

        self.conv_1 = nn.Conv2d(2*self.attention_channels[0], self.attention_channels[0], 1)
        self.upconv_1 = nn.ConvTranspose2d(self.attention_channels[0], self.attention_channels[0], kernel_size=4, stride=4, padding=0, bias=False)

        # self._init_weights()

    def _init_weights(self):
            """Custom weight initialization to ensure positivity."""
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    torch.nn.init.uniform_(m.weight, 0.00, 0.1)  # Positive values between 0.01 and 1.0
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.00)  # Ensure positive bias

    def process_stage(self, features_left, features_right, stage_idx, N, H, W,  features_left_up=None, features_right_up=None, upconv=None, conv=None):
        """Process a single stage of the disparity estimation pipeline"""
        if upconv is not None and conv is not None and features_left_up is not None and features_right_up is not None:
            # Upsample and concatenate features from previous stage
            left_up = upconv(features_left_up)
            right_up = upconv(features_right_up)
            
            left = torch.cat([left_up, features_left], dim=1)
            right = torch.cat([right_up, features_right], dim=1)
            
            left = conv(left)
            right = conv(right)
        else:
            left = features_left
            right = features_right

        # Compute correlation and cost volume
        corr_method = AGCL(left, right, self.corr_split_mode[stage_idx])
        corr_res, _ = corr_method(self.group_wise_split[stage_idx], self.search_num[stage_idx])
        
        # Reshape correlation volume
        corr_seq = torch.reshape(
            torch.permute(corr_res, (0,2,3,1)),
            (N, H*W, self.search_num[stage_idx]*self.group_wise_split[stage_idx]),
        )
        
        # Process through attention and linear layers
        cost_seq = self.att_encs[stage_idx](corr_seq, corr_seq, source_mask=None)
        cost_seq = self.nn_linears[stage_idx](cost_seq)

        # Generate attention weights and compute disparity
        attn_weights = F.sigmoid(cost_seq, dim=-1)
        attn_weights = torch.reshape(
            torch.permute(attn_weights, (0,2,3,1)),
            (N, H*W, 17),
        )
        
        return attn_weights

    def forward(self, features_left, features_right):
        attn_weight = []
        for i in range(4):
            N, C, H, W = features_left[i].size()

            features_left[i] = self.self_conv_fn[i](features_left[i])
            features_right[i] = self.self_conv_fn[i](features_right[i])

            position_encoding = PositionEncodingSine(d_model=self.attention_channels[i], max_shape=(features_left[i].size(2), features_left[i].size(3)))
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

            features_left[i], features_right[i] = self.cross_att_fn[i](features_left[i], features_right[i])

            features_left[i], features_right[i] = [
                torch.permute(torch.reshape(x, (N, H, W, C)), (0,3,1,2)) #(N, C, H, W)
                for x in [features_left[i], features_right[i]]
            ]

            features_left[i] = self.self_conv_fn2[i](features_left[i])
            features_right[i] = self.self_conv_fn2[i](features_right[i])    


        # Stage 4
        N, C, H, W = features_left[3].size()
        attn_weight4 = self.process_stage(features_left[3], features_right[3], 3, N, H, W)
        attn_weight.append(attn_weight4)

        # Stage 3
        H, W = H*2, W*2
        attn_weight3 = self.process_stage(features_left[2], features_right[2], 2, N, H, W,
                                    features_left[3], features_right[3],
                                    self.upconv_4, self.conv_3)
        attn_weight.append(attn_weight3)

        # Stage 2
        H, W = H*2, W*2
        attn_weight2 = self.process_stage(features_left[1], features_right[1], 1, N, H, W,
                                    features_left[2], features_right[2],
                                    self.upconv_3, self.conv_2)
        attn_weight.append(attn_weight2)

        # Stage 1
        H, W = H*2, W*2
        attn_weight1 = self.process_stage(features_left[0], features_right[0], 0, N, H, W,
                                    features_left[1], features_right[1],
                                    self.upconv_2, self.conv_1)
        attn_weight.append(attn_weight1)

        # Final stage
        H, W = H*4, W*4
        vol_0_left = self.upconv_1(features_left[0])
        vol_0_right = self.upconv_1(features_right[0])
        attn_weight_final = self.process_stage(vol_0_left, vol_0_right, 0, N, H, W)
        attn_weight.append(attn_weight_final)

        return attn_weight

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