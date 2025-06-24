import os
import torch.utils.data as torch_data
import numpy as np
import random
from PIL import Image
from .dataset_template import DatasetTemplate


class ManualDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.load(disp_img_path).astype(np.float32)
        # for validation, full resolution disp need to be divided by 128 instead of 256

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img
        }
        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
