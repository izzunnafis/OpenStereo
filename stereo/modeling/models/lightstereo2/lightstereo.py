import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation
from .submodule import local_variance_filter
from .aggregation import MobileV2Residual
import time


class LightStereo2(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCBONE', 'MobileNetv2'))

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
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

        #freq filter
        self.freq_filter = nn.Sequential(
            MobileV2Residual(3, 16, stride=2, expanse_ratio=4),
            MobileV2Residual(16, 48, stride=2, expanse_ratio=4),
            MobileV2Residual(48, 48, stride=1, expanse_ratio=4),
            MobileV2Residual(48, 48, stride=1, expanse_ratio=4))



    def forward(self, data):
        # time_start = time.time()
        image1 = data['left']
        image2 = data['right']
        # torch.cuda.synchronize()
        # print('time_start:', time_start - time.time())

        # time1 = time.time()
        features_left = self.backbone(image1)
        features_right = self.backbone(image2)
        # torch.cuda.synchronize()
        # print("time_backbone:", time.time() - time1)
        
        filter = local_variance_filter(image1, 5)
        freq_filter_left = self.freq_filter(filter)

        # time2 = time.time()
        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        # torch.cuda.synchronize()
        # print("time_correlation_volume:", time.time() - time2)

        # time3 = time.time()
        encoding_volume = self.cost_agg(gwc_volume, features_left, freq_filter_left)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]
        # torch.cuda.synchronize()
        # print("time_cost_agg:", time.time() - time3)

        # time4 = time.time()
        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]
        # torch.cuda.synchronize()
        # print("time_disp_regression:", time.time() - time4)

        # time5 = time.time()
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]
        # torch.cuda.synchronize()
        # print("time_refine:", time.time() - time5)

        # time6 = time.time()
        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4
        # torch.cuda.synchronize()
        # print("time_result:", time.time() - time6)
        # print("total_time:", time.time() - time_start)

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
