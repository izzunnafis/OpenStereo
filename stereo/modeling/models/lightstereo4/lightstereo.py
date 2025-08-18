import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume, build_concat_volume, build_gwc_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation
from .submodule import local_variance_filter, edge_detection, LearnableVarianceFilter, LearnableCorrelationVolume
from .aggregation import MobileV2Residual
import time


class LightStereo4(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCBONE', 'MobileNetv2'))

        self.img_texture = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 48, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU)
        )

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
        self.loss = LogL1Loss_v3()

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        corr_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        gwc_vol = build_gwc_volume(features_left[-1], features_right[-1], self.max_disp // 16, num_groups=8)

        encoding_volume = self.cost_agg(corr_volume, vol_3d=gwc_vol)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]

        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        # time5 = time.time()
        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'disp_pred': disp_pred}
        result['filter'] = filter

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
        # loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')
        loss = 1.0 * self.loss(disp_pred[mask], disp_gt[mask])

        disp_4 = model_pred['disp_4']
        disp_4 = torch.clamp(disp_4, min=1e-6, max=self.max_disp)
        if torch.isnan(disp_4).any() or torch.isinf(disp_4).any():
            print('disp_4 has nan or inf')
            disp_4 = torch.nan_to_num(disp_4, nan=1e-6, posinf=self.max_disp, neginf=1e-6)
        # loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')
        loss += 0.3 * self.loss(disp_pred[mask], disp_gt[mask])  # Adding the loss for the initial prediction


        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print('loss has nan or inf')
            loss = torch.nan_to_num(loss, nan=0, posinf=100, neginf=0)

        loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info
    
class LogL1Loss(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self):
        super(LogL1Loss, self).__init__()
        self.crit = nn.L1Loss()

    def forward(self, f_s, f_t):
        f_s= torch.log(f_s + 0.5)
        f_t= torch.log(f_t + 0.5)
        loss = self.crit(f_s, f_t)
        return loss
    
class LogL1Loss_v2(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self):
        super(LogL1Loss_v2, self).__init__()
        self.crit = nn.L1Loss(reduction=None)

    def forward(self, f_s, f_t):
        loss = torch.log(torch.abs(f_s - f_t) + 1.0).mean()
        return loss
 
class LogL1Loss_v3(nn.Module):
    """
    Loss to apply at student vs teacher predictions and/or student vs GT (replacing smoothL1)
    """
    def __init__(self, beta=2.71828, epsilon=1.0):
        super(LogL1Loss_v3, self).__init__()
        self.crit = nn.L1Loss(reduction=None)
        self.beta = torch.tensor(beta)
        self.epsilon = torch.tensor(epsilon)

    def forward(self, f_s, f_t):
        diff = torch.abs(f_s-f_t)
        loss = torch.where(
            diff < self.beta,
            torch.log(diff + self.epsilon),
            diff/(self.beta+self.epsilon) + torch.log(self.beta + self.epsilon) - self.beta/(self.beta+self.epsilon)
        )
        return loss.mean()
    
class MobileV2ResidualMod(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, dilation=1):
        super(MobileV2ResidualMod, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expanse_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        pad = dilation

        # v2
        self.pwconv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, pad, dilation=dilation, groups=hidden_dim, bias=False),
        )
        self.pwliner = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
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
