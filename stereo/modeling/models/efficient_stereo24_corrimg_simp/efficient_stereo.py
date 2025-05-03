import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from stereo.modeling.models.efficient_stereo24_corrimg_simp.backbone import Backbone

# from .backbone import Backbone
from .aggregation import Aggregation, FPNLayer


class EfficientStereo24(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(backbone="MobileNetv2")

        # aggregation
        self.cost_agg = Aggregation(input_channel=self.backbone.out_channel)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.out_channel[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)


    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        encoded_vol = self.cost_agg(features_left, features_right, image1, image2) # [bz, self.attention_channels[0], H/4, W/4]

        prob_vol = F.softmax(encoded_vol, dim=1)
        init_disp = disparity_regression(prob_vol, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'disp_pred' : disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
        
        return result


    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        disp_pred = torch.clamp(disp_pred, min=1e-6, max=self.max_disp)
        if torch.isnan(disp_pred).any() or torch.isinf(disp_pred).any():
            print('disp_pred has nan or inf')
            disp_pred = torch.nan_to_num(disp_pred, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')

        disp_4 = model_pred['disp_4']
        disp_4 = torch.clamp(disp_4, min=1e-6, max=self.max_disp)
        if torch.isnan(disp_4).any() or torch.isinf(disp_4).any():
            print('disp_4 has nan or inf')
            disp_4 = torch.nan_to_num(disp_4, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print('loss has nan or inf')
            loss = torch.nan_to_num(loss, nan=0, posinf=100, neginf=0)

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info


    def encode_disp_gt(self, disp_gt, disp_lists):
        """
        Encode disparity ground truth into a binary vector representation.
        
        Args:
            disp_gt (torch.Tensor): [bz, 1, h, w], containing disparity values.
            disp_lists (torch.Tensor): [num_levels], containing possible disparity values.
        
        Returns:
            torch.Tensor: Encoded disparity in shape [bz, num_levels, h, w]
        """
        bz, _, h, w = disp_gt.shape
        disp_lists = disp_lists.to(disp_gt.device).view(1, -1, 1, 1)  # [1, num_levels, 1, 1]
        
        # Create an empty tensor to store the encoded values
        encoded_disp = torch.zeros(bz, len(disp_lists[0]), h, w, device=disp_gt.device)
        
        # Iterate through the bits and set values where disparity matches
        remaining_disp = disp_gt.clone()
        for i in range(len(disp_lists[0])):
            encoded_disp[:, i, :, :] = (remaining_disp >= disp_lists[0][i]).float().view(bz, h, w)
            remaining_disp -= encoded_disp[:, i, :, :].view(bz, 1, h, w) * disp_lists[0][i]

        return encoded_disp