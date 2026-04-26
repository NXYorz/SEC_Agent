#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: NXY
"""

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from kaiwudrl.interface.agent import BaseAgent
from agent_diy.model.model import Model
from agent_diy.conf.conf import Config
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.definition import ActData, ObsData
import random
import numpy as np

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)
    
    def observation_process(self, obs):
        """
        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
        函数的输入：
            - obs: 环境返回的局部观测信息
            - preprocessor: 预处理器
            - extra_info: 环境返回的全局状态信息
        函数的输出：
            - ObsData: 用于模型推理的观测数据
            - remain_info: 用于奖励计算的其他数据
        """
        feature, legal_action, reward = self.preprocessor.feature_process(obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        res = []
        for i in range(len(list_obs_data)):
            feature = list_obs_data[i].feature
            legal_action = list_obs_data[i].legal_action

            logits, value, prob = self._run_model(feature, legal_action)
            # 训练阶段保持“采样分布 == 学习分布”，避免 PPO 比率失真导致难以收敛。
            prob = self._apply_action_heuristics(
                np.array(prob, dtype=np.float32), feature, legal_action
            )
            prob = self._normalize_probs(prob)

            action = self._legal_sample(prob, use_max=False)
            d_action = self._legal_sample(prob, use_max=True)

            res.append(ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            ))

        return res

    def exploit(self, env_obs):
        """
        评估时贪心选择动作（利用）。
        """
        #同理，这里act_data看上去是个列表，实际上只有一个元素
        obs_data, _ = self.observation_process(env_obs)
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        _, value, prob = self._run_model(feature, legal_action)
        prob = self._apply_action_heuristics(np.array(prob, dtype=np.float32), feature, legal_action)
        d_action = self._legal_sample(prob, use_max=True)
        act_data = ActData(
            action=[d_action],
            d_action=[d_action],
            prob=list(prob),
            value=value,
        )
        return self.action_process(act_data, is_stochastic=False)


    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """
        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    

    def action_process(self, act_data, is_stochastic=True):
        """
        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])
    
    def reset(self, env_obs=None):
        """
        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def _run_model(self, feature, legal_action):
        """
        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """
        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """
        按概率分布采样动作。
        """
        probs = self._normalize_probs(probs)
        if use_max:
            return int(np.argmax(probs))
        # 避免 np.random.multinomial 在浮点极小误差下抛出
        # "sum(pvals[:-1]) > 1.0" 的数值异常，改用手写 CDF 采样。
        cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
        if cdf.size == 0:
            return 0
        cdf[-1] = 1.0
        r = float(np.random.random())
        return int(np.searchsorted(cdf, r, side="right"))

    def _apply_action_heuristics(self, prob, feature, legal_action):
        """
        轻量规则后处理：减少原地打转/卡脚，并在安全时提升吃箱倾向。
        """
        legal = np.array(legal_action, dtype=np.float32)
        f = np.array(feature, dtype=np.float32)
        is_stop = f[6] > 0.5
        is_cycle = f[7] > 0.5
        is_danger = f[8] > 0.5
        box_alive = f[10] > 0.5
        box_dir = int(f[11]) if f[11] >= 0 else -1

        # map_info 衍生特征: 邻格可走性 + 通路深度，用于路线优化
        map_next = f[42:50]   # 每个方向下一格是否可通行
        map_depth = f[50:58]  # 每个方向在局部视野内可连续前进深度(0~1)

        # 先做一层通行约束引导：
        # - 邻格阻塞动作降权，减少“撞墙/贴边抖动”
        # - 深通路动作增权，优先选择更开阔路线
        for d in range(8):
            if legal[d] <= 0.5:
                continue
            if map_next[d] <= 0.5:
                prob[d] *= 0.08
                prob[d + 8] *= 0.20
            else:
                prob[d] *= (0.70 + 0.90 * float(map_depth[d]))

        # 常规巡航期抑制“折线游走”：轻微延续上一步方向，减少无意义急转。
        if (not is_danger) and (not is_stop) and (not is_cycle) and 0 <= self.last_action < 8:
            last_dir = self.last_action
            if legal[last_dir] > 0.5:
                prob[last_dir] *= 1.25
            # 大角度拐弯（±3、反向）在安全期降权，避免蛇形/折线路径拖慢逃生。
            prob[(last_dir + 3) % 8] *= 0.78
            prob[(last_dir + 5) % 8] *= 0.78
            prob[(last_dir + 4) % 8] *= 0.70

            # 非危险期默认压低闪现偏好，避免“无效位移”干扰路径学习。
            prob[8:16] *= 0.45

        # 卡脚时，提高普通移动动作占比，并降低闪现动作占比（除非处于危险）。
        if is_stop or is_cycle:
            prob[0:8] *= 1.35
            if not is_danger:
                prob[8:16] *= 0.08

            if 0 <= self.last_action < 8:
                opp = (self.last_action + 4) % 8
                # 避免继续顶墙，同时显式鼓励反向脱困。
                prob[self.last_action] *= 0.02 if is_stop else 0.10
                if legal[opp] > 0.5:
                    prob[opp] *= 4.00 if is_stop else 2.80
                # 轻微提升与反向相邻的两个方向，降低“原地打转”概率。
                prob[(opp + 1) % 8] *= 2.00 if is_stop else 1.60
                prob[(opp + 7) % 8] *= 2.00 if is_stop else 1.60

            # 结合局部地图，优先选择“可走且通路更深”的脱困方向。
            legal_move = legal[:8] > 0.5
            open_score = np.where(legal_move, map_next * (0.5 + map_depth), -1.0)
            best_dir = int(np.argmax(open_score)) if np.max(open_score) > 0 else -1
            if best_dir >= 0:
                prob[best_dir] *= 2.60 if is_stop else 1.80
                prob[(best_dir + 1) % 8] *= 1.18
                prob[(best_dir + 7) % 8] *= 1.18

        # 安全状态下，若有宝箱则优先朝宝箱方向移动，减少“无意义游走”。
        if (not is_danger) and box_alive and 0 <= box_dir < 8 and legal[box_dir] > 0.5:
            prob[box_dir] *= 2.4
            prob[(box_dir + 1) % 8] *= 1.15
            prob[(box_dir + 7) % 8] *= 1.15

        # 强危险时，基于最近怪物方向进行“规避引导”。
        if is_danger:
            near_m1 = f[20]  # monster1 dist norm
            near_m2 = f[25]  # monster2 dist norm
            near_idx = 0 if near_m1 <= near_m2 else 1

            mx = f[17] if near_idx == 0 else f[22]
            mz = f[18] if near_idx == 0 else f[23]
            mvis = f[16] if near_idx == 0 else f[21]
            if mvis > 0.5:
                hx, hz = f[0], f[1]
                danger_dir = self._vector_to_dir(mx - hx, mz - hz)
                escape_dir = (danger_dir + 4) % 8
                if legal[danger_dir] > 0.5:
                    prob[danger_dir] *= 0.35
                    prob[(danger_dir + 1) % 8] *= 0.60
                    prob[(danger_dir + 7) % 8] *= 0.60
                if legal[escape_dir] > 0.5:
                    base_gain = 2.20 if min(near_m1, near_m2) < 0.25 else 1.55
                    prob[escape_dir] *= base_gain
                    prob[(escape_dir + 1) % 8] *= 1.20
                    prob[(escape_dir + 7) % 8] *= 1.20

                # 高危时适度提高闪现逃生动作权重（对应移动方向 +8）。
                if min(near_m1, near_m2) < 0.18 and legal[escape_dir + 8] > 0.5:
                    prob[escape_dir + 8] *= 1.45

        prob = prob * legal
        s = float(prob.sum())
        if s <= 1e-8:
            valid = np.where(legal > 0.5)[0]
            if len(valid) == 0:
                return np.ones_like(prob) / len(prob)
            prob = np.zeros_like(prob)
            prob[valid] = 1.0 / len(valid)
            return prob
        return self._normalize_probs(prob)

    def _vector_to_dir(self, dx, dz):
        """Convert relative vector into 8-way direction index."""
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return 0
        ang = float(np.arctan2(dz, dx))
        # map [-pi, pi] to [0, 8)
        idx = int(np.round(((ang + np.pi) / (2 * np.pi)) * 8.0)) % 8
        return idx

    def _normalize_probs(self, probs):
        """
        将概率向量稳定归一化到有效分布，避免 multinomial 因浮点误差报错。
        """
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p.size == 0:
            return p

        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.clip(p, 0.0, None)

        s = float(p.sum())
        if s <= 0.0:
            p = np.ones_like(p, dtype=np.float64) / float(p.size)
            return p.astype(np.float32)

        p = p / s

        # 再次归一，确保 float64 检查时前 n-1 项和不会因舍入超过 1。
        p = p / float(p.sum())
        if p.size > 1:
            tail = 1.0 - float(p[:-1].sum(dtype=np.float64))
            p[-1] = np.clip(tail, 0.0, 1.0)

        s2 = float(p.sum(dtype=np.float64))
        if s2 <= 0.0:
            p = np.ones_like(p, dtype=np.float64) / float(p.size)
        else:
            p = p / s2

        return p.astype(np.float32)
