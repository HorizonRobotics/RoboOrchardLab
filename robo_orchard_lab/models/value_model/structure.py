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

import logging

import torch

from robo_orchard_lab.models.holobrain import (
    HoloBrain_Qwen2_5_VL,
    HoloBrain_Qwen2_5_VLConfig,
)
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
)

__all__ = ["HoloBrain_Value_Qwen2_5_VL", "HoloBrain_Value_Qwen2_5_VLConfig"]

logger = logging.getLogger(__name__)

MODULE_TPYE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class HoloBrain_Value_Qwen2_5_VL(HoloBrain_Qwen2_5_VL):  # noqa: N801
    def __init__(self, cfg: "HoloBrain_Qwen2_5_VLConfig"):
        super().__init__(cfg)

    def loss(self, inputs):
        pred_logits, _, _, _ = self._forward(inputs)

        loss = {}
        loss["pred_value_loss"] = self.decoder.loss(pred_logits, inputs)
        return loss

    @torch.no_grad()
    def predict(self, inputs):
        pred_logits, _, _ = self._forward(inputs)
        if self.decoder.out_dim == 1:
            pred_logits = pred_logits.squeeze(-1)  # bs, 64
            pred_value = -torch.sigmoid(
                pred_logits
            )  # bs, 64; gt value is in range [-1,0]
        if self.decoder.out_dim > 1:
            pred_probs = torch.softmax(pred_logits, dim=-1)
            pred_value = self.decoder.loss.transform_from_probs(
                pred_probs
            )  # convert bin to value
        return pred_value

    def load_state_dict(self, state_dict, strict=False, **kwargs):

        # filter out state keys with tensor shape mismatch manually
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in self.state_dict() and v.shape == self.state_dict()[k].shape
        }

        if self.use_state_dict_with_vlm:
            return super().load_state_dict(
                filtered_state_dict, strict=strict, **kwargs
            )

        filtered_state_dict = {
            k: v
            for k, v in filtered_state_dict.items()
            if not k.startswith("vlm.")
        }
        incompatible_keys = super().load_state_dict(
            filtered_state_dict, strict=False, **kwargs
        )
        missing_keys = []
        for key in incompatible_keys.missing_keys:
            if not key.startswith("vlm."):
                missing_keys.append(key)
        incompatible_keys = type(incompatible_keys)(
            missing_keys, incompatible_keys.unexpected_keys
        )

        # reinitialize states of missing_keys
        param_dict = dict(self.named_parameters())
        for k in missing_keys:
            if k not in param_dict:
                raise KeyError(f"{k} not found in model parameters")
            param = param_dict[k]
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.zeros_(param)

        if strict:
            assert (
                len(incompatible_keys.missing_keys) == 0
                and len(incompatible_keys.unexpected_keys) == 0
            ), (
                "Unexpected key(s) in state_dict: {}. Missing key(s) in state_dict: {}.".format(  # noqa: E501
                    ", ".join(
                        f'"{k}"' for k in incompatible_keys.unexpected_keys
                    ),
                    ", ".join(
                        f'"{k}"' for k in incompatible_keys.missing_keys
                    ),
                )
            )
        return incompatible_keys


class HoloBrain_Value_Qwen2_5_VLConfig(HoloBrain_Qwen2_5_VLConfig):  # noqa: N801
    class_type: ClassType_co[HoloBrain_Value_Qwen2_5_VL] = (
        HoloBrain_Value_Qwen2_5_VL
    )
