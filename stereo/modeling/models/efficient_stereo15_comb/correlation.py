# Adopted from https://github.com/megvii-research/CREStereo/blob/master/nets/corr.py

import numpy as np
import torch
import torch.nn.functional as F

import torch.nn as nn
import math

def bilinear_sampler(img, coords):
    """Wrapper for grid_sample, uses pixel coordinates"""

    img = F.remap(img, coords, border_mode="constant")

    return img


def coords_grid(batch, ht, wd):
    x_grid, y_grid = np.meshgrid(np.arange(wd), np.arange(ht))
    y_grid, x_grid = torch.tensor(y_grid, dtype="float32"), torch.tensor(
        x_grid, dtype="float32"
    )
    coords = F.stack([x_grid, y_grid], axis=0)
    coords = F.repeat(F.expand_dims(coords, axis=0), batch, axis=0)
    return coords

class AGCL:
    """
    Implementation of Adaptive Group Correlation Layer (AGCL).
    """

    def __init__(self, fmap1, fmap2, split_mode=0):
        self.fmap1 = fmap1
        self.fmap2 = fmap2

        self.split_mode = split_mode

    def __call__(self, split_num=4, search_num=9):
        corr = self.corr_iter(self.fmap1, self.fmap2, split_num, search_num)
        return corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = int(psize[0] // 2 * di_y), int((psize[1]-1) * di_x)

        right_pad = F.pad(right_feature, pad=(padx, 0, pady, pady), mode="replicate")

        right_slid = right_pad.unfold(2, H, di_y).unfold(3, W, di_x)
        right_slid = right_slid.reshape(N, C, -1, H, W)
        left_feature = torch.unsqueeze(left_feature, axis=2)

        corr_mean = torch.mean(left_feature * right_slid, axis=1) #N, num_select, H, W

        return corr_mean

    def corr_iter(self, left_feature, right_feature, split_num=4, search_num=25):
        assert int(np.sqrt(search_num)) ** 2 == search_num, "search_num must be a perfect square"

        if self.split_mode == 0:
            psize_list = [(np.sqrt(search_num), np.sqrt(search_num))]*split_num
            dilate_list = [(1, 1)]*split_num

            # Create 2D disparity grid
            offset_x, offset_y = torch.meshgrid(
                torch.arange(0, np.sqrt(search_num), 1),
                torch.arange(-(np.sqrt(search_num) // 2), (np.sqrt(search_num) // 2) + 1, 1),
                indexing="ij"
            )

            # Flatten to (D, 2) where D = 25
            disparity_offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=1)

            # Compute L2 distance
            disp_lists = torch.norm(disparity_offsets.float(), dim=1)

            # Create copies of disp_lists
            disp_lists = disp_lists.repeat(1, split_num)
            

        elif self.split_mode == 1:
            psize_list = [(1, search_num)]*split_num
            dilate_list = [(1, 1)]*split_num

            disp_lists = torch.arange(0, search_num, 1).repeat(1, split_num)


        lefts = torch.split(left_feature, split_num, dim=1)
        rights = torch.split(right_feature, split_num, dim=1)

        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(
                lefts[i], rights[i], psize_list[i], dilate_list[i]
            )
            corrs.append(corr)

        final_corr = torch.concat(corrs, dim=1)

        return final_corr, disp_lists
    

class CorrAsAttention(nn.Module):
    """
    Correlation-As-Attention (C-A²) module for stereo matching.

    Args:
        in_channels (int): Number of input feature channels from backbone.
        heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        max_disp (int): Maximum disparity (number of shifts) to consider.
    """
    def __init__(self, in_channels, heads=4, head_dim=16, max_disp=64):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.head_dim = head_dim
        self.max_disp = max_disp
        self.inner_dim = head_dim * heads

        # Linear projections for Q, K, V
        self.to_q = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1, bias=False)

        # Output projection
        self.to_out = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)

    def forward(self, left_feat, right_feat):
        """
        Forward pass.

        Args:
            left_feat: [B, C, H, W] feature map from left image
            right_feat: [B, C, H, W] feature map from right image

        Returns:
            out: [B, C, H, W] aggregated feature map
            disp_map: [B, H, W] disparity map via soft-argmax
        """
        B, C, H, W = left_feat.shape

        # Compute queries
        q = self.to_q(left_feat).view(B, self.heads, self.head_dim, H * W)  # [B, heads, dim, P]

        # Build keys and values for each disparity shift
        k_list, v_list = [], []
        for d in range(self.max_disp):
            if d > 0:
                # shift right-features to the right by d pixels
                shifted = F.pad(right_feat[:, :, :, :-d], (d, 0, 0, 0), mode='constant', value=0)
            else:
                shifted = right_feat
            k = self.to_k(shifted).view(B, self.heads, self.head_dim, H * W)
            v = self.to_v(shifted).view(B, self.heads, self.head_dim, H * W)
            k_list.append(k)
            v_list.append(v)
        # Stack disparities: [B, heads, dim, D, P]
        k = torch.stack(k_list, dim=3)
        v = torch.stack(v_list, dim=3)

        # Compute attention scores: [B, heads, P, D]
        scores = torch.einsum('b h c p, b h c d p -> b h p d', q, k)
        scores = scores / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # Aggregate values: [B, heads, dim, P]
        out = torch.einsum('b h p d, b h c d p -> b h c p', attn, v)
        out = out.contiguous().view(B, self.inner_dim, H, W)

        # Project back to original channel space
        out = self.to_out(out)

        # # Disparity regression via soft-argmax
        # attn_mean = attn.mean(dim=1)  # [B, P, D]
        # disp_values = torch.arange(self.max_disp, device=left_feat.device).view(1, 1, self.max_disp)
        # disp_map = (attn_mean * disp_values).sum(dim=2).view(B, H, W)

        return out
