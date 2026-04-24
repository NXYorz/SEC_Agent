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
        #NXY:其实也可以直接list_obs_data[0]，因为输入是形如[obs_data]这样的
        self.logger.info(f"[NXY DEBUG2]:lst_obs_data =  {list_obs_data}")
        self.logger.info(f"[NXY DEBUG2]: len(list_obs_data) = {len(list_obs_data)}")
        res = []
        for i in range(len(list_obs_data)):
            self.logger.info(f"[NXY DEBUG3]: entry loop successfully")
            feature = list_obs_data[i].feature
            legal_action = list_obs_data[i].legal_action

            logits, value, prob = self._run_model(feature, legal_action)

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
        act_data = self.predict([obs_data])
        stochastic = random.randint(0,1)
        if stochastic == 1:
            is_stochastic = True
        else:
            is_stochastic = False
        return self.action_process(act_data[0], is_stochastic)


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
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
