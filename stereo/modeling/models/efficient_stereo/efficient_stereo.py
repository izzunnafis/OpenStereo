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

        self.disp_lists = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625])
        self.disp_lists = self.disp_lists.to("cuda")

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        attn_weight = self.cost_agg(features_left, features_right) # [bz, hw, 18]

        out = []
        tmp_scale = [32, 16, 8, 4, 1]
        N, H, W = image1.size()[0], 320, 736
        for i in range(len(attn_weight)):
            disp_val = torch.sum(attn_weight[i] * self.disp_lists, axis=-1)
            out.append(disp_val.reshape(N, 1, H//tmp_scale[i], W//tmp_scale[i]))
            attn_weight[i] = torch.permute(
                                torch.reshape(attn_weight[i], (N, H//tmp_scale[i], W//tmp_scale[i], 18)),
                                (0,3,1,2),
                            )

        result = {'disp_pred' : out[-1],
                  'disp_per4' : out[-2],
                  'disp_per8' : out[-3],
                  'disp_per16' : out[-4],
                  'disp_per32' : out[-5],
                  'attn_weight_pred': attn_weight[-1],
                    'attn_weight_per4': attn_weight[-2],
                    'attn_weight_per8': attn_weight[-3],
                    'attn_weight_per16': attn_weight[-4],
                    'attn_weight_per32': attn_weight[-5]
                  }
        
        return result


    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]
        mask = mask.repeat(1, len(self.disp_lists), 1, 1)  # [bz, num_levels, h, w]

        disp_pred = model_pred['attn_weight_pred']
        disp_pred = torch.clamp(disp_pred, min=1e-6, max=self.max_disp)
        if torch.isnan(disp_pred).any() or torch.isinf(disp_pred).any():
            print('disp_pred has nan or inf')
            disp_pred = torch.nan_to_num(disp_pred, nan=1e-6, posinf=self.max_disp, neginf=1e-6)

        disp_gt_encoded = self.encode_disp_gt(disp_gt, self.disp_lists)

        print(model_pred['attn_weight_pred'][0, :, 100, 0])
        print(disp_gt_encoded[0, :, 100, 0])

        loss_final = F.mse_loss(disp_pred[mask], disp_gt_encoded[mask], reduction='mean')
        loss = 1.0 * loss_final

        # disp_per4 = model_pred['disp_per4']
        # disp_per4 = torch.clamp(disp_per4, min=1e-6, max=self.max_disp)
        # disp_gt_per4 = F.interpolate(disp_gt, scale_factor=0.25, mode='bilinear', align_corners=False)
        # mask =  (disp_gt_per4 < self.max_disp) & (disp_gt_per4 > 0)
        # if torch.isnan(disp_per4).any() or torch.isinf(disp_per4).any():
        #     print('disp_per4 has nan or inf')
        #     disp_per4 = torch.nan_to_num(disp_per4, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        # loss_per4 = F.smooth_l1_loss(disp_per4[mask], disp_gt_per4[mask], reduction='mean')
        # # loss += 0.3 * loss_per4

        # disp_per8 = model_pred['disp_per8']
        # disp_per8 = torch.clamp(disp_per8, min=1e-6, max=self.max_disp)
        # disp_gt_per8 = F.interpolate(disp_gt, scale_factor=0.125, mode='bilinear', align_corners=False)
        # mask =  (disp_gt_per8 < self.max_disp) & (disp_gt_per8 > 0)
        # if torch.isnan(disp_per8).any() or torch.isinf(disp_per8).any():
        #     print('disp_per8 has nan or inf')
        #     disp_per8 = torch.nan_to_num(disp_per8, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        # loss_per8 = F.smooth_l1_loss(disp_per8[mask], disp_gt_per8[mask], reduction='mean')
        # # loss += 0.3 * loss_per8

        # disp_per16 = model_pred['disp_per16']
        # disp_per16 = torch.clamp(disp_per16, min=1e-6, max=self.max_disp)
        # disp_gt_per16 = F.interpolate(disp_gt, scale_factor=0.0625, mode='bilinear', align_corners=False)
        # mask =  (disp_gt_per16 < self.max_disp) & (disp_gt_per16 > 0)
        # if torch.isnan(disp_per16).any() or torch.isinf(disp_per16).any():
        #     print('disp_per16 has nan or inf')
        #     disp_per16 = torch.nan_to_num(disp_per16, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        # loss_per16 = F.smooth_l1_loss(disp_per16[mask], disp_gt_per16[mask], reduction='mean')
        # # loss += 0.3 * loss_per16

        # disp_per32 = model_pred['disp_per32']
        # disp_per32 = torch.clamp(disp_per32, min=1e-6, max=self.max_disp)
        # disp_gt_per32 = F.interpolate(disp_gt, scale_factor=0.03125, mode='bilinear', align_corners=False)
        # mask =  (disp_gt_per32 < self.max_disp) & (disp_gt_per32 > 0)
        # if torch.isnan(disp_per32).any() or torch.isinf(disp_per32).any():
        #     print('disp_per32 has nan or inf')
        #     disp_per32 = torch.nan_to_num(disp_per32, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        # loss_per32 = F.smooth_l1_loss(disp_per32[mask], disp_gt_per32[mask], reduction='mean')
        # # loss += 0.3 * loss_per32

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print('loss has nan or inf')
            loss = torch.nan_to_num(loss, nan=0, posinf=100, neginf=0)

        loss_info = {'scalar/train/loss_disp': loss.item()}
                    #  'scalar/train/loss_disp_final': loss_final.item(),
                    #  'scalar/train/loss_disp_per4': loss_per4.item(),
                    #  'scalar/train/loss_disp_per8': loss_per8.item(),
                    #  'scalar/train/loss_disp_per16': loss_per16.item(),
                    #  'scalar/train/loss_disp_per32': loss_per32.item()}

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