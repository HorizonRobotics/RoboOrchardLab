# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn


class HoloBrainValueLoss(nn.Module):
    def __init__(
        self, min_value, max_value, num_bins, loss_mode="twohot", sigma=1.0
    ):
        super().__init__()
        self.loss_mode = loss_mode
        assert self.loss_mode in ["l2", "onehot", "twohot", "hlgauss"]
        if self.loss_mode != "l2":
            self.min_value = min_value
            self.max_value = max_value
            self.num_bins = num_bins
            self.sigma = sigma * (max_value - min_value) / num_bins
            self.register_buffer(
                "support",
                torch.linspace(
                    min_value, max_value, num_bins + 1, dtype=torch.float32
                ),
            )

    def l2loss(self, logits, target, pred_mask):
        logits = logits.squeeze(-1)  # bs, 64
        pred = -torch.sigmoid(logits)  # bs, 64; gt value is in range [-1,0]
        if (
            pred.dim() == 2 and target.dim() == 1
        ):  # pred has an additional dim of pred_steps
            target = target[:, None]
        loss = (pred - target) ** 2
        if pred_mask is not None:
            loss = loss[pred_mask]
        loss = loss.mean()
        return loss

    def cross_entropy_loss(self, logits, target_probs, pred_mask):
        if logits.dim() == 3 and target_probs.dim() == 2:
            target_probs = target_probs[:, None, :].expand(
                -1, logits.shape[1], -1
            )

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_probs * log_probs)
        if pred_mask is not None:
            loss = loss[pred_mask]
        loss = loss.sum(dim=-1).mean()
        return loss

    def forward(self, logits, inputs, **kwargs):
        target = inputs["value"]
        pred_mask = inputs["pred_mask"]
        if self.loss_mode == "l2":
            loss = self.l2loss(logits, target, pred_mask)
        else:
            if self.loss_mode == "onehot":
                target_probs = self._transform_to_probs_onehot(target)
            elif self.loss_mode == "twohot":
                target_probs = self._transform_to_probs_twohot(target)
            elif self.loss_mode == "hlgauss":
                target_probs = self._transform_to_probs_hlgauss(target)
            loss = self.cross_entropy_loss(logits, target_probs, pred_mask)
        return loss

    def _transform_to_probs_onehot(self, target):
        """One-Hot Hard Label."""
        # 映射到 [0, num_bins-1] 坐标系
        target = torch.clamp(target, self.min_value, self.max_value)
        norm_target = (target - self.min_value) / (
            self.max_value - self.min_value
        )
        bin_idx = (
            (norm_target * (self.num_bins - 1)).round().long()
        )  # 使用 round 比 floor 更准确对齐最近的 Bin
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        return F.one_hot(bin_idx, num_classes=self.num_bins).float()

    def _transform_to_probs_twohot(self, target):
        """Two-Hot (Linear Interpolation)."""
        target = target.float()
        target = torch.clamp(target, self.min_value, self.max_value)
        # 映射到连续坐标 [0, num_bins - 1]
        coords = (
            (target - self.min_value)
            / (self.max_value - self.min_value)
            * (self.num_bins - 1)
        )

        idx_left = coords.floor().long()
        idx_right = torch.clamp(idx_left + 1, max=self.num_bins - 1)

        weight_right = coords - idx_left.float()
        weight_left = 1.0 - weight_right

        # 创建输出容器
        batch_idx = torch.arange(target.size(0), device=target.device)
        output = torch.zeros(
            target.size(0), self.num_bins, device=target.device
        )

        output[batch_idx, idx_left] = weight_left
        output[batch_idx, idx_right] += weight_right
        return output

    def _transform_to_probs_hlgauss(self, target):
        """HL-Gauss: Histogram Loss with Gaussian Targets."""
        # 1. 构建高斯 CDF
        target = target.float()
        target = target.unsqueeze(-1)

        # 标准正态分布 CDF: 0.5 * (1 + erf(x / sqrt(2)))
        # self.support 已经在 GPU 上 (如果模型在 GPU)
        scaled_diff = (self.support - target) / (
            self.sigma * 1.41421356
        )  # sqrt(2) approx
        cdf_evals = torch.special.erf(scaled_diff)

        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        return bin_probs / (z.unsqueeze(-1) + 1e-8)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # 计算 Bin 的中心点
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)
