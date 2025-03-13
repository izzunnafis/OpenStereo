import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone
from .aggregation import Aggregation


class EfficientStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(cfgs.get('input_size', '[3, 1280, 352]'))

        # aggregation
        self.cost_agg = Aggregation()


    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        out = self.cost_agg(features_left, features_right)

        result = {'disp_pred' : out[-1],
                  'disp_per4' : out[-2],
                  'disp_per8' : out[-3],
                  'disp_per16' : out[-4],
                  'disp_per32' : out[-5]}
        return result


    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        disp_pred = torch.clamp(disp_pred, min=1e-6, max=self.max_disp)
        print(disp_pred[mask])
        print(disp_gt[mask])
        if torch.isnan(disp_pred).any() or torch.isinf(disp_pred).any():
            print('disp_pred has nan or inf')
            disp_pred = torch.nan_to_num(disp_pred, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
                
        loss_final = F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')
        loss = 1.0 * loss_final

        disp_per4 = model_pred['disp_per4']
        disp_per4 = torch.clamp(disp_per4, min=1e-6, max=self.max_disp)
        disp_gt_per4 = F.interpolate(disp_gt, scale_factor=0.25, mode='bilinear', align_corners=False)
        mask =  (disp_gt_per4 < self.max_disp) & (disp_gt_per4 > 0)
        if torch.isnan(disp_per4).any() or torch.isinf(disp_per4).any():
            print('disp_per4 has nan or inf')
            disp_per4 = torch.nan_to_num(disp_per4, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        loss_per4 = F.smooth_l1_loss(disp_per4[mask], disp_gt_per4[mask], reduction='mean')
        # loss += 0.3 * loss_per4

        disp_per8 = model_pred['disp_per8']
        disp_per8 = torch.clamp(disp_per8, min=1e-6, max=self.max_disp)
        disp_gt_per8 = F.interpolate(disp_gt, scale_factor=0.125, mode='bilinear', align_corners=False)
        mask =  (disp_gt_per8 < self.max_disp) & (disp_gt_per8 > 0)
        if torch.isnan(disp_per8).any() or torch.isinf(disp_per8).any():
            print('disp_per8 has nan or inf')
            disp_per8 = torch.nan_to_num(disp_per8, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        loss_per8 = F.smooth_l1_loss(disp_per8[mask], disp_gt_per8[mask], reduction='mean')
        # loss += 0.3 * loss_per8

        disp_per16 = model_pred['disp_per16']
        disp_per16 = torch.clamp(disp_per16, min=1e-6, max=self.max_disp)
        disp_gt_per16 = F.interpolate(disp_gt, scale_factor=0.0625, mode='bilinear', align_corners=False)
        mask =  (disp_gt_per16 < self.max_disp) & (disp_gt_per16 > 0)
        if torch.isnan(disp_per16).any() or torch.isinf(disp_per16).any():
            print('disp_per16 has nan or inf')
            disp_per16 = torch.nan_to_num(disp_per16, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        loss_per16 = F.smooth_l1_loss(disp_per16[mask], disp_gt_per16[mask], reduction='mean')
        # loss += 0.3 * loss_per16

        disp_per32 = model_pred['disp_per32']
        disp_per32 = torch.clamp(disp_per32, min=1e-6, max=self.max_disp)
        disp_gt_per32 = F.interpolate(disp_gt, scale_factor=0.03125, mode='bilinear', align_corners=False)
        mask =  (disp_gt_per32 < self.max_disp) & (disp_gt_per32 > 0)
        if torch.isnan(disp_per32).any() or torch.isinf(disp_per32).any():
            print('disp_per32 has nan or inf')
            disp_per32 = torch.nan_to_num(disp_per32, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        loss_per32 = F.smooth_l1_loss(disp_per32[mask], disp_gt_per32[mask], reduction='mean')
        # loss += 0.3 * loss_per32

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print('loss has nan or inf')
            loss = torch.nan_to_num(loss, nan=0, posinf=100, neginf=0)

        loss_info = {'scalar/train/loss_disp': loss.item(),
                     'scalar/train/loss_disp_final': loss_final.item(),
                     'scalar/train/loss_disp_per4': loss_per4.item(),
                     'scalar/train/loss_disp_per8': loss_per8.item(),
                     'scalar/train/loss_disp_per16': loss_per16.item(),
                     'scalar/train/loss_disp_per32': loss_per32.item()}

        return loss, loss_info
