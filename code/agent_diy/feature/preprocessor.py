#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np
from agent_diy.feature.definition import reward_shaping

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _to_mask8(raw):
    """Convert diverse legal-action payloads to 8D move mask."""
    mask = None
    if isinstance(raw, (list, tuple, np.ndarray)):
        data = list(raw)
        # nested payload
        if len(data) > 0 and isinstance(data[0], (list, tuple, np.ndarray)):
            return _to_mask8(data[0])

        if len(data) > 0 and all(isinstance(x, (bool, int, np.bool_, np.integer)) for x in data):
            # 明确的 8 维 0/1 掩码
            if len(data) >= 8 and all(int(x) in (0, 1) for x in data[:8]):
                mask = [int(x) for x in data[:8]]
            # 任意长度的方向下标列表（例如 [0,2,5]）
            elif all(0 <= int(x) < 8 for x in data):
                valid = {int(x) for x in data}
                mask = [1 if i in valid else 0 for i in range(8)]

    if isinstance(raw, dict):
        for key in ("move_dir", "move", "move_action", "direction", "legal_move"):
            if key in raw:
                return _to_mask8(raw[key])

    return mask if mask is not None else [1] * 8


def _to_mask16(raw):
    """Convert legal-action payloads to 16D mask: [move8, flash8]."""
    if isinstance(raw, (list, tuple, np.ndarray)):
        data = list(raw)
        if len(data) > 0 and all(isinstance(x, (bool, int, np.bool_, np.integer)) for x in data):
            # 明确的 16 维 0/1 掩码
            if len(data) >= 16 and all(int(x) in (0, 1) for x in data[:16]):
                return [int(x) for x in data[:16]]

            # 任意长度动作下标列表（例如 [0,3,8,11]）
            if all(0 <= int(x) < 16 for x in data):
                valid = {int(x) for x in data}
                return [1 if i in valid else 0 for i in range(16)]

        if len(data) >= 2 and isinstance(data[0], (list, tuple, np.ndarray)) and isinstance(
            data[1], (list, tuple, np.ndarray)
        ):
            move = _to_mask8(data[0])
            flash = _to_mask8(data[1])
            return move + flash

    if isinstance(raw, dict):
        move = None
        flash = None
        for key in ("move_dir", "move", "move_action", "direction", "legal_move"):
            if key in raw:
                move = _to_mask8(raw[key])
                break
        for key in ("flash_dir", "flash", "flash_action", "legal_flash"):
            if key in raw:
                flash = _to_mask8(raw[key])
                break
        if move is not None or flash is not None:
            move = move if move is not None else [1] * 8
            # flash 缺失时默认全禁用，避免被误解析为“全部可闪现”
            flash = flash if flash is not None else [0] * 8
            return move + flash

    move = _to_mask8(raw)
    # 无明确 flash 信息时，默认禁用 flash，防止采样到环境非法动作导致“原地不动”。
    return move + [0] * 8



def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def get_area(x , z):
    return (int)(x / 8) , (int)(z / 8)

def Dis(monster , hero):
    return np.sqrt((monster["pos"]["x"] - hero["pos"]["x"]) ** 2 + (monster["pos"]["z"] - hero["pos"]["z"]) ** 2)

#危险返回True
def check_monsterAndHero(monster , hero , env_info):
    """Danger judgement for a single monster.

    统一按真实距离与怪物速度判断危险，避免依赖闪现 CD 导致误判。
    """
    if monster is None or len(monster) == 0:
        return False

    if float(monster.get("is_in_view", 1)) <= 0:
        return False

    dist = float(Dis(monster, hero))
    speed = float(monster.get("speed", 1.0))
    flash_cd = float(hero.get("flash_cooldown", 0.0))

    # 闪现不可用时，危险半径更大；怪物速度越高，危险半径越大。
    danger_radius = 30.0 + min(10.0, speed * 2.0)
    if flash_cd > 0:
        danger_radius += 10.0

    return dist <= danger_radius

def check_monstersAndhero(monsters , hero , env_info):
    if len(monsters) == 0:
        return 0
    isDanger1 = check_monsterAndHero(monsters[0] , hero , env_info)
    isDanger2 = False
    if len(monsters) > 1:
        isDanger2 = check_monsterAndHero(monsters[1] , hero , env_info)
    if isDanger1 == True or isDanger2 == True:
        return 1
    return 0

def Greddy(isDanger , monsters , organs):
    if isDanger == 0:
        return 1
    if len(monsters) == 0:
        return 1
    if len(organs) == 0:
        return 0
    isNotEuqal1 = True
    if organs[0]["hero_relative_direction"] == monsters[0]["hero_relative_direction"]:
        isNotEuqal1 = False
    isNotEuqal12 = True
    if len(monsters) > 1:
        if organs[0]["hero_relative_direction"] == monsters[1]["hero_relative_direction"]:
            isNotEuqal12 = False
    if isNotEuqal12 == True and isNotEuqal1 == True:
        return 1
    return 0

def check_box(box , monsters):
    if len(monsters) == 0:
        return 0
    box_x , box_z = get_area(box["pos"]["x"] , box["pos"]["z"])
    monster1_x , monster1_z = get_area(monsters[0]["pos"]["x"] , monsters[0]["pos"]["z"])
    monster2_x = box_x
    monster2_z = box_z
    if len(monsters) > 1:
        monster2_x , monster2_z = get_area(monsters[1]["pos"]["x"] , monsters[1]["pos"]["z"])
    if box_x == monster1_x and box_z == monster1_z:
        return 1
    if box_x == monster2_x and box_z == monster2_z:
        return 1
    return 0

class Preprocessor:
    def __init__(self):
        self.reset()
        self.last_x = -1
        self.last_z = -1

        self.last_area_x = 0
        self.last_area_z = 0
        self.cycleStep = 0

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.cycleStep = 0
        self.reward_state = {
            "last_box_score": 0.0,
            "last_survive_score": 0.0,
            "last_total_score": 0.0,
            "last_box_dist_norm": 1.0,
            "last_min_monster_dist_norm": 0.5,
            "last_flash_cooldown": 0.0,
            "last_is_stop": 0.0,
            "last_is_cycle": 0.0,
        }

    

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (10D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        # flash_ready=1 表示“可用”，避免语义与字段名相反。
        flash_ready = 1.0 if float(hero.get("flash_cooldown", 0.0)) <= 0.0 else 0.0
        buff_remain_norm = _norm(env_info["buff_refresh_time"], MAX_BUFF_DURATION)
        # 分数/步数做归一化，避免大数值特征压制其它输入。
        score = np.tanh(float(env_info.get("total_score", 0.0)) / 100.0)
        frame_id = _norm(env_info.get("step_no", 0), self.max_step)
        isStop = 0
        if hero_pos["x"] == self.last_x and hero_pos["z"] == self.last_z:
            isStop = 1
        self.last_x = hero_pos["x"]
        self.last_z = hero_pos["z"]

        area_x , area_z = get_area(hero_pos["x"] , hero_pos["z"])
        if area_x == self.last_area_x and area_z == self.last_area_z:
            self.cycleStep += 1
        else:
            self.cycleStep = 0
            self.last_area_x = area_x
            self.last_area_z = area_z
        isCycle = 0
        if self.cycleStep > 8:
            isCycle = 1
        isDanger = check_monstersAndhero(frame_state.get("monsters", []) , hero , env_info)
        isGredy = Greddy(isDanger , frame_state.get("monsters", []) , frame_state.get("organs", []))
        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_ready, buff_remain_norm , score , frame_id , isStop , isCycle, isDanger , isGredy], dtype=np.float32)

        #宝箱特征 6D
        isEffect = 0
        direction = 0
        dis = 0
        isBoxDanger = 0
        box_x = 0
        box_z = 0
        box = []
        if len(frame_state.get("organs", [])) > 0:
            box = frame_state.get("organs", [])[0]
            isEffect = box["status"]
            direction = box["hero_relative_direction"]
            dis = _norm(Dis(box , hero), MAP_SIZE * 1.41)
            isBoxDanger = check_box(box , frame_state.get("monsters", []))
            box_x = _norm(box["pos"]["x"], MAP_SIZE)
            box_z = _norm(box["pos"]["z"], MAP_SIZE)
        box_feat = np.array([isEffect , direction , dis , isBoxDanger , box_x , box_z] , dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                # 当怪物不存在时，距离应为“最远”（1.0），否则奖励会误判为“怪物贴脸”。
                monster_feats.append(np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

        # Legal action mask (16D) / 合法动作掩码
        legal_action = _to_mask16(legal_act_raw)

        if sum(legal_action[:8]) == 0:
            # 防止“全非法 -> 全动作放开”导致频繁采样到无效闪现，改为仅放开移动动作。
            legal_action = [1] * 8 + [0] * 8
        flash_cooldown = hero["flash_cooldown"]
        for i in range(8,16):
            if flash_cooldown > 0:
                legal_action[i] = 0
            elif legal_action[i] not in (0, 1):
                legal_action[i] = legal_action[i - 8]

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                box_feat,
                monster_feats[0],
                monster_feats[1],
                np.array(legal_action, dtype=np.float32),
                map_feat,
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        frame_no = env_info["step_no"]
        reward = reward_shaping(self , frame_no, hero, monsters , box , monster_feats, hero_feat , env_info)
        return feature, legal_action, reward
