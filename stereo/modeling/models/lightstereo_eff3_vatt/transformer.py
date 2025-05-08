"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""


import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import math


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.unsqueeze(torch.cumsum(torch.ones(max_shape), 0), 0)
        x_position = torch.unsqueeze(torch.cumsum(torch.ones(max_shape), 1), 0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2) * (-math.log(10000.0) / d_model // 2)
        )
        div_term = torch.unsqueeze(div_term, 1).unsqueeze(2)  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.pe = torch.unsqueeze(pe, 0)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.shape[2], : x.shape[3]].to(x.device)
    




def elu(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

def elu_feature_map(x):
    return elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * torch.unsqueeze(q_mask, 2).unsqueeze(3)  # [:, :, None, None]
        if kv_mask is not None:
            K = K * torch.unsqueeze(kv_mask, 2).unsqueeze(3)  # [:, :, None, None]
            values = values * torch.unsqueeze(kv_mask, 2).unsqueeze(3)  # [:, :, None, None]

        v_length = values.shape[1]
        values = values / v_length  # prevent fp16 overflow
        KV = torch.sum(torch.unsqueeze(K, -1) * torch.unsqueeze(values, 3), dim=1)
        Z = 1 / (torch.sum(Q * torch.sum(K, dim=1, keepdim=True), dim=-1) + self.eps)
        queried_values = (
            torch.sum(
                torch.unsqueeze(Q, -1) * torch.unsqueeze(KV, 1) * torch.unsqueeze(Z, 3).unsqueeze(4),
                dim=3,
            )
            * v_length
        )

        return queried_values


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(drop_prob=attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.sum(torch.unsqueeze(queries, 2) * torch.unsqueeze(keys, 1), dim=-1)
        if kv_mask is not None:
            assert q_mask.dtype == np.bool_
            assert kv_mask.dtype == np.bool_
            QK[
                ~(torch.unsqueeze(q_mask, 2).unsqueeze(3) & torch.unsqueeze(kv_mask, 1).unsqueeze(3))
            ] = float("-inf")

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.shape[3] ** 0.5  # sqrt(D)
        A = F.softmax(softmax_temp * QK, axis=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.sum(torch.unsqueeze(A, -1) * torch.unsqueeze(values, 1), dim=2)

        return queried_values
    
"""
Transformer module proposed in "LoFTR: Detector-Free Local Feature Matching with Transformers"
Modified from: https://github.com/zju3dv/LoFTR/tree/master/src/loftr
"""


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention="linear"):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False).cuda()
        self.k_proj = nn.Linear(d_model, d_model, bias=False).cuda()
        self.v_proj = nn.Linear(d_model, d_model, bias=False).cuda()
        self.attention = LinearAttention() if attention == "linear" else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False).cuda()

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        ).cuda()

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model).cuda()
        self.norm2 = nn.LayerNorm(d_model).cuda()

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.shape[0]
        query, key, value = x, source, source

        # multi-head attention
        query = torch.reshape(
            self.q_proj(query), (bs, -1, self.nhead, self.dim)
        )  # [N, L, (H, D)] (H=8, D=256//8)
        key = torch.reshape(
            self.k_proj(key), (bs, -1, self.nhead, self.dim)
        )  # [N, S, (H, D)]
        value = torch.reshape(self.v_proj(value), (bs, -1, self.nhead, self.dim))
        message = self.attention(
            query, key, value, q_mask=x_mask, kv_mask=source_mask
        )  # [N, L, (H, D)]
        message = self.merge(
            torch.reshape(message, (bs, -1, self.nhead * self.dim))
        )  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.concat([x, message], axis=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model, nhead, layer_names, attention):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = LoFTREncoderLayer(d_model, nhead, attention)
        self.layers = [
            copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))
        ]
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.ndim > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert (
            self.d_model == feat0.shape[2]
        ), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == "self":
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == "cross":
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            elif name == "cross_single":
                feat0 = layer(feat0, feat1, mask0, mask1)   
            else:
                raise KeyError

        return feat0, feat1