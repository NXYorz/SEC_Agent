#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

峡谷追猎 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import os
import time

import torch
import torch.nn.functional as F
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        """
        训练入口：对一批 SampleData 执行 PPO 更新。
        """
        obs = torch.stack([f.obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_actions for f in list_sample_data]).to(self.device)
        act = torch.stack([f.actions for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.probs for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantages for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.values for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.rewards for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "reward": round(reward.mean().item(), 4),
            }
            self.logger.info(
                f"[train] total_loss:{results['total_loss']} "
                f"policy_loss:{results['policy_loss']} "
                f"value_loss:{results['value_loss']} "
                f"entropy:{results['entropy_loss']}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """
        计算标准 PPO 损失（策略损失 + 价值损失 + 熵正则化）。
        """
        # Masked logits / 合法动作掩码后的 logits
        masked_logits = self._masked_logits(logits, legal_action)
        log_prob_dist = F.log_softmax(masked_logits, dim=1)
        prob_dist = log_prob_dist.exp()

        # Policy loss (PPO Clip) / 策略损失
        one_hot = F.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_log_prob = (one_hot * log_prob_dist).sum(1, keepdim=True)
        # 行为策略概率有时会非常接近 0（数值噪声/掩码边界），
        # 直接参与 ratio 会让 policy_loss 大幅抖动，这里提高下界抑制尖峰梯度。
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-5)
        old_log_prob = old_action_prob.log()
        ratio = (new_log_prob - old_log_prob).exp().clamp(0.0, 10.0)
        adv = advantage.view(-1, 1)
        adv_std = adv.std(unbiased=False)
        if float(adv_std) > 1e-3:
            adv = (adv - adv.mean()) / (adv_std + 1e-8)
        else:
            # 小方差 batch 不做标准化，避免噪声被 1/std 放大。
            adv = adv - adv.mean()
        adv = adv.clamp(-5.0, 5.0)
        policy_loss1 = ratio * adv
        policy_loss2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = -torch.minimum(policy_loss1, policy_loss2).mean()

        # Value loss (Clipped) / 价值损失
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                torch.square(tdret - vp),
                torch.square(tdret - value_clip),
            ).mean()
        )

        # Entropy loss / 熵损失
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1).mean()

        # Total loss / 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, [value_loss, policy_loss, entropy_loss]

    def _masked_logits(self, logits, legal_action):
        """
        合法动作掩码下的 logits（将非法动作置为极小值）。
        """
        legal_action = (legal_action > 0.5).float()
        masked_logits = logits.masked_fill(legal_action < 0.5, -1e10)

        # 若某行全非法，则只放开前 8 个普通移动动作，避免训练中梯度落在无效动作上。
        valid_count = legal_action.sum(dim=1, keepdim=True)
        bad_row = valid_count < 0.5
        if bad_row.any():
            fallback_mask = torch.zeros_like(legal_action)
            fallback_mask[:, : min(8, fallback_mask.size(1))] = 1.0
            legal_action = torch.where(bad_row, fallback_mask, legal_action)
            masked_logits = logits.masked_fill(legal_action < 0.5, -1e10)
        return masked_logits
