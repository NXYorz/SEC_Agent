#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: NXY
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from agent_diy.conf.conf import Config

def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc

"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out = out + identity
        out = F.silu(out)
        return out
"""

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)

class Model(nn.Module):
    """
    输入:
        feature_vec: [B, 60]
        feature_img: [B, 4, 51, 51]

    输出:
        value:  [B, 1]
        logits: [B, 16]   (如果 softmax=True，则返回概率)
    """
    def __init__(self, device=None, softmax=False):
        super().__init__()

        # 基本配置
        self.model_name = "sec"
        #self.VIEW_SIZE = 50
        self.FEATURES = [10, 6, 10, 16, 16, 2]
        self.FEATURE_LEN = sum(self.FEATURES)  # 60
        #self.IMAGE_SHAPE = (4, self.VIEW_SIZE + 1, self.VIEW_SIZE + 1)  # (4, 51, 51)
        self.ACTION_NUM = 16
        self.VALUE_NUM = 1
        self.softmax = softmax

        self.device = device

        # 特征分段索引
        # [10, 6, 10, 16, 16, 2]
        self.hero_start, self.hero_end = 0, 10
        self.box_start, self.box_end = 10, 16
        self.monster_start, self.monster_end = 16, 26
        self.mask_start, self.mask_end = 26, 42
        self.local_start, self.local_end = 42, 58
        self.progress_start, self.progress_end = 58, 60

        # 图像分支 CNN
        # 输入: [B, 4, 60, 60]
        # 输出: [B, 256]
        """
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            ResidualBlock(32, 64, stride=2),   # 51 -> 26
            ResidualBlock(64, 128, stride=2),  # 26 -> 13
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2), # 13 -> 7
            ResidualBlock(256, 256, stride=1),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()  # [B, 256]
        )
        """

        # 向量分支：按语义分别编码
        self.hero_encoder = MLPBlock(10, 32, 32, dropout=0.05)
        self.box_encoder = MLPBlock(6, 16, 16, dropout=0.05)
        self.monster_encoder = MLPBlock(10, 32, 32, dropout=0.05)
        self.mask_encoder = MLPBlock(16, 32, 32, dropout=0.00)
        self.local_encoder = MLPBlock(16, 32, 32, dropout=0.05)
        self.progress_encoder = MLPBlock(2, 8, 8, dropout=0.00)

        self.backbone = nn.Sequential(
            nn.Linear(32 + 16 + 32 + 32 + 32 + 8, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )

        # 融合层
        # 图像256 + 向量128 => 256
        """
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.10),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        """

        # 双头输出
        # policy: 8维动作 logits
        # value : 1维状态价值
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, self.ACTION_NUM)
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, self.VALUE_NUM)
        )

        self._init_weights()
        #self.to(self.device)

    def _init_weights(self):
        """
        比较适合 RL 场景的初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # policy head 最后一层常用较小初始化
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)

        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def encode_vector(self, feature_vec):
        """
        feature_vec: [B, 60]
        """
        hero_feat = self.hero_encoder(feature_vec[:, self.hero_start:self.hero_end])
        box_feat = self.box_encoder(feature_vec[:, self.box_start:self.box_end])
        monster_feat = self.monster_encoder(feature_vec[:, self.monster_start:self.monster_end])
        mask_feat = self.mask_encoder(feature_vec[:, self.mask_start:self.mask_end])
        local_feat = self.local_encoder(feature_vec[:, self.local_start:self.local_end])
        progress_feat = self.progress_encoder(feature_vec[:, self.progress_start:self.progress_end])

        x = torch.cat(
            [hero_feat, box_feat, monster_feat, mask_feat, local_feat, progress_feat],
            dim=1
        )
        return self.backbone(x)

    def forward(self, feature_vec, use_action_mask=True):
        """
        参数:
            feature_vec: [B, 60] 或 [60]
            use_action_mask: 是否对非法动作做mask

        返回:
            value, logits_or_probs
        """
        if feature_vec.dim() == 1:
            feature_vec = feature_vec.unsqueeze(0)

        feature_vec = feature_vec.to(self.device).float()
        #feature_img = feature_img.to(self.device).float()

        # 图像分支
        #img_feat = self.image_encoder(feature_img)   # [B, 256]

        # 融合
        fused = self.encode_vector(feature_vec)
        
        # 双头
        logits = self.policy_head(fused)  # [B, 16]
        value = self.value_head(fused)    # [B, 1]

        # 动作掩码：FEATURE 中第4段是 16 维合法动作掩码
        if use_action_mask:
            action_mask = feature_vec[:, self.mask_start:self.mask_end]  # [B, 16]
            valid = action_mask > 0.5
            logits = logits.masked_fill(~valid, -1e9)

        if self.softmax:
            policy = torch.softmax(logits, dim=-1)
            return value, policy
        else:
            return logits , value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
