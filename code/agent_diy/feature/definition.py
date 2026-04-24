#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: NXY
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None

# ObsData: feature=50D vector, legal_action=8D mask / 特征向量与合法动作掩码
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
)

# ActData: action, d_action(greedy), prob, value / 动作、贪心动作、概率、价值
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None, 
    prob=None, 
    value=None,
)


# SampleData用于在aisrv和learner之间传递训练样本
# 必须使用整数定义维度（不能用None），框架层会自动生成FIELD_DIMS并处理序列化
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # 观测维度
    legal_actions=16,  # 合法动作维度
    actions=1,  # 动作维度（标量）
    probs=16,  # 动作概率分布维度
    reward=1,  # 奖励（标量）
    rewards=1,  # 奖励（标量）
    advantages=1,  # 优势函数（标量）
    values=1,  # 价值函数（标量）
    dones=1,  # 是否结束（标量）
    next_value=1,
)

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0

SURVIVE_REWARD = 0.8
TREASURE_REWARD = 1.0
BOX_REWARD = 0.1
MONSTER_REWARD = 0.5
FAR_MONSTER_REWARD = 0.5


last_box_score = 0
last_survive_score = 0
last_box_dis = 0
last_mostMonster_dis = 0
last_farMonster_dis = 0

def Dis(monster , hero):
    if len(monster) == 0:
        return 1
    return np.sqrt((monster["pos"]["x"] - hero["pos"]["x"]) ** 2 + (monster["pos"]["z"] - hero["pos"]["z"]) ** 2)

def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0



def reward_shaping(preprocessor , frame_no, hero, monsters , box , monster_feats , hero_feat , env):
    cur_monst_dist_norm1 = monster_feats[0][4]
    cur_monst_dist_norm2 = monster_feats[1][4]       
    
    reward = 0
    #生存奖励
    global last_survive_score
    if(last_survive_score < env["step_score"]):
        reward += SURVIVE_REWARD
    last_survive_score = env["step_score"]

    #宝箱奖励
    global last_box_score
    if(last_box_score < env["treasure_score"]):
        reward += TREASURE_REWARD
    last_box_score = env["treasure_score"]

    #宝箱接近奖励
    global last_box_dis
    cur_box_dist_norm = _norm(Dis(box , hero) , MAP_SIZE * 1.41)
    if last_box_dis < cur_box_dist_norm:
        reward += BOX_REWARD
    last_box_dis = cur_box_dist_norm
    
    #怪物远离奖励
    cur_monst_min_dis = min(cur_monst_dist_norm1 , cur_monst_dist_norm2)
    global last_mostMonster_dis
    if last_mostMonster_dis < cur_monst_min_dis:
        reward += MONSTER_REWARD
    else:
        reward -= MONSTER_REWARD  
    last_mostMonster_dis = cur_monst_min_dis

    #第二只怪物压力奖励
    far_monster_dis = max(cur_monst_dist_norm1 , cur_monst_dist_norm2)
    global last_farMonster_dis
    if last_farMonster_dis < far_monster_dis:
        reward += FAR_MONSTER_REWARD
    last_farMonster_dis = far_monster_dis

    return reward

def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].values

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.values + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.values
