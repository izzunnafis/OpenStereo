# @Time    : 2024/3/10 10:21
# @Author  : zhangchenming

import torch
import torch.nn as nn
import timm

import time


class Backbone(nn.Module):
    def __init__(self, backbone=''):
        super().__init__()

        if backbone == 'MobileNetv2':
            out_block = [1, 2, 4]
            out_channel = [24, 32, 96]
            mobilenet2_model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)

            stem = nn.Sequential(
                mobilenet2_model.conv_stem,
                mobilenet2_model.bn1,
            )
            blocks = mobilenet2_model.blocks[0:5]
        elif backbone == 'MobileNetv4':
            out_block = [0, 1, 2]
            out_channel = [48, 96, 192]
            mobilenet4_model = timm.create_model('mobilenetv4_conv_large', pretrained=True, features_only=True)
            stem = nn.Sequential(
                mobilenet4_model.conv_stem,
                mobilenet4_model.bn1,
                mobilenet4_model.act1,
            )
            blocks = mobilenet4_model.blocks[0:3]
        elif backbone == 'ResNet34':
            out_block = [0, 1, 2]
            out_channel = [64, 128, 256]
            resnet34_model = timm.create_model('resnet34', pretrained=True, features_only=True)
            stem = nn.Sequential(
                resnet34_model.conv1,
                resnet34_model.bn1,
                resnet34_model.act1,
                resnet34_model.maxpool
            )
            blocks = nn.Sequential(
                resnet34_model.layer1,
                resnet34_model.layer2,
                resnet34_model.layer3
            )
        else:
            raise NotImplementedError

        self.out_block = out_block
        self.out_channel = out_channel

        self.stem = stem
        self.blocks = blocks

    def forward(self, images):
        out_features = []
        x = self.stem(images)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_block:
                out_features.append(x)

        return out_features


if __name__ == "__main__":
    model = Backbone('ResNet34').cuda()
    model2 = Backbone('MobileNetv4').cuda()
    model3 = Backbone('MobileNetv2').cuda()
    dummy_input = torch.randn(1, 3, 1024, 1024).cuda()  # Example input tensor

    for i in range(100):
        start_time = time.time()
        features3 = model(dummy_input)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        print(f"ResNet34 Inference Time: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        features2 = model2(dummy_input)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        print(f"MobileNetv4 Inference Time: {time.time() - start_time:.6f} seconds")

        start_time = time.time()
        features = model3(dummy_input)
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        print(f"MobileNetv2 Inference Time: {time.time() - start_time:.6f} seconds")
