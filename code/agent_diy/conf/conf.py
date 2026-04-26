#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: NXY
"""


import numpy as np

# Configuration
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

   
    #特征向量，50D
    FEATURES = [
        10,#英雄主特征,当前X坐标，Y坐标、闪现是否可用，BUFF,当前分数进展、、已经进入第几个阶段、最近是否卡住、最近是否在重复绕路，是否危险，该贪还是保
        6,#宝箱特征,宝箱是否还有效、和自己相对方向、相对距离、当前危险度，位置
        10,#怪物特征,是否可见，位置，速度，距离，dist_norm(两只怪物)
        16,#合法动作掩码(往哪里走更安全)
        16,#局部地图特征
        2,#进度特征
    ]
    # Whether to use CNN networks
    # 是否使用CNN网络
    USE_CNN = True
    VIEW_SIZE = 50 if USE_CNN else 0

    FEATURE_VECTOR_SHAPE = FEATURES
    FEATURE_IMAGE_SHAPE = (4, VIEW_SIZE + 1, VIEW_SIZE + 1)

    FEATURE_LEN = sum(FEATURE_VECTOR_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    #GAMMA = 0.95

    # Initial learning rate
    # 初始的学习率
    #START_LR = 5e-4

    #VALUE_LOSS_COEFF = 0.5
    #ENTROPY_LOSS_COEFF = 0.025

    # Action space / 动作空间：16个移动方向
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
    # PPO 每批样本的重复优化轮次；过小会导致 policy_loss 长期贴近 0 难以更新。
    PPO_EPOCHS = 4
