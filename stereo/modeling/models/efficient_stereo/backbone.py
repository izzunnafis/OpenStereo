# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming

from transformers import AutoModel
from timm.data.transforms_factory import create_transform

import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self, backbone='', size=(3, 320, 736)):
        super().__init__()
        self.model = AutoModel.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
        self.channels = [80, 160, 320, 640]

        self.transform = create_transform(input_size=size,
                                     is_training=True, 
                                     mean=self.model.config.mean,
                                     std=self.model.config.std,
                                     crop_mode=self.model.config.crop_mode,
                                     crop_pct=self.model.config.crop_pct)

    def forward(self, images):
        images = self.transform(images)
        outputs, features = self.model(images)

        # return features [B, 80, H/4, W/4], [B, 160, H/8, W/8], [B, 320, H/16, W/16], [B, 640, H/32, W/32]

        return features

if __name__ == "__main__":
    model = Backbone(size=[3, 256, 256]).cuda()
    dummy_input = torch.randn(1, 3, 512, 512).cuda()  # Example input tensor

    features = model(dummy_input)
    for out in features:
        print(out[0].size())
