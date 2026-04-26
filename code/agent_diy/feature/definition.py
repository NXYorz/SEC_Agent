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

# Reward coefficients
# 奖励权重：降低“苟活”收益，增强“找箱子并拿到箱子”的收益
SURVIVE_REWARD = 0.16
TREASURE_REWARD = 2.2
BOX_APPROACH_REWARD = 0.35
BOX_LEAVE_PENALTY = -0.12
MONSTER_ESCAPE_REWARD = 0.45
MONSTER_TOO_CLOSE_PENALTY = -0.75
MONSTER_APPROACH_PENALTY = -0.35
IDLE_PENALTY = -0.16
CYCLE_PENALTY = -0.10
EARLY_FLASH_PENALTY = -0.2
UNSTUCK_REWARD = 0.0
SCORE_GAIN_REWARD = 0.06

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
    # 仅统计“可见怪物”的距离，避免无怪物时被错误判定为“怪物贴脸”。
    visible_monster_dist = [float(mf[4]) for mf in monster_feats if float(mf[0]) > 0.5]
    if len(visible_monster_dist) == 0:
        cur_monst_min_dis = 1.0
    else:
        cur_monst_min_dis = min(visible_monster_dist)

    reward = 0.0
    rs = preprocessor.reward_state

    # 生存奖励：保留，但显著降低，避免学到“原地苟步数”
    if rs["last_survive_score"] < env["step_score"]:
        reward += SURVIVE_REWARD
    rs["last_survive_score"] = env["step_score"]

    # 宝箱奖励：主目标，显著提高
    if rs["last_box_score"] < env["treasure_score"]:
        reward += TREASURE_REWARD
    rs["last_box_score"] = env["treasure_score"]

    # 让训练目标与监控总分同向：总分上涨就给正奖励。
    cur_total_score = float(env.get("total_score", 0.0))
    last_total_score = float(rs.get("last_total_score", 0.0))
    delta_total_score = cur_total_score - last_total_score
    if delta_total_score > 0:
        reward += SCORE_GAIN_REWARD * delta_total_score
    rs["last_total_score"] = cur_total_score

    # 宝箱距离 shaping：接近加分，远离扣分（修复原先符号方向）
    if len(box) != 0:
        cur_box_dist_norm = _norm(Dis(box , hero) , MAP_SIZE * 1.41)
        if cur_box_dist_norm < rs["last_box_dist_norm"]:
            reward += BOX_APPROACH_REWARD * (rs["last_box_dist_norm"] - cur_box_dist_norm)
        else:
            reward += BOX_LEAVE_PENALTY * (cur_box_dist_norm - rs["last_box_dist_norm"])
        rs["last_box_dist_norm"] = cur_box_dist_norm

    # 危险规避 shaping：近距离怪物时鼓励拉开，过近则直接惩罚
    if cur_monst_min_dis < 0.20:
        reward += MONSTER_TOO_CLOSE_PENALTY
    elif cur_monst_min_dis > rs["last_min_monster_dist_norm"]:
        reward += MONSTER_ESCAPE_REWARD * (cur_monst_min_dis - rs["last_min_monster_dist_norm"])
    elif cur_monst_min_dis < rs["last_min_monster_dist_norm"] and cur_monst_min_dis < 0.35:
        reward += MONSTER_APPROACH_PENALTY * (rs["last_min_monster_dist_norm"] - cur_monst_min_dis)
    rs["last_min_monster_dist_norm"] = cur_monst_min_dis

   # 反“打转摆烂”：停滞与循环区域惩罚
    is_stop = hero_feat[6]
    is_cycle = hero_feat[7]
    if is_stop > 0.5:
        reward += IDLE_PENALTY
    if is_cycle > 0.5:
        reward += CYCLE_PENALTY
    # 从停滞/绕圈恢复时给予小额正反馈，帮助学习“及时换向”
    if rs.get("last_is_stop", 0.0) > 0.5 and is_stop <= 0.5:
        reward += UNSTUCK_REWARD
    if rs.get("last_is_cycle", 0.0) > 0.5 and is_cycle <= 0.5:
        reward += UNSTUCK_REWARD
    rs["last_is_stop"] = float(is_stop)
    rs["last_is_cycle"] = float(is_cycle)

    # 闪现误用惩罚：开局/非危险状态滥用闪现扣分
    now_flash_cd = hero["flash_cooldown"]
    last_flash_cd = rs["last_flash_cooldown"]
    is_danger = hero_feat[8] > 0.5
    if now_flash_cd > last_flash_cd and not is_danger and frame_no < 80:
        reward += EARLY_FLASH_PENALTY
    rs["last_flash_cooldown"] = now_flash_cd

    return float(np.clip(reward, -1.0, 1.5))

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
        done = float(sample.dones[0]) if hasattr(sample, "dones") else 0.0
        not_done = 1.0 - done
        delta = -sample.values + sample.reward + gamma * sample.next_value * not_done
        gae = gae * gamma * lamda + delta
        # Keep field names aligned with SampleData definitions used by learner.
        sample.advantages = gae.astype(np.float32)
        sample.rewards = (gae + sample.values).astype(np.float32)
