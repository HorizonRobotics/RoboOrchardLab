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
from torch import nn

from robo_orchard_lab.models.holobrain import (
    HoloBrainActionDecoder,
    UpsampleHead,
)
from robo_orchard_lab.models.holobrain.utils import (
    apply_scale_shift,
    recompute,
)

logger = logging.getLogger(__name__)


class ValueUpsampleHead(UpsampleHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        bs, num_joint, num_chunk, state_dims = x.shape
        # we don't need the num_joint dim anymore for value prediction
        x = x.mean(dim=1)  # bs, num_chunk, state_dims
        for i, layer in enumerate(self.upsamples):
            if i in self.norm_act_idx:
                x = self.act_and_norm[i](x)
            x = x.permute(0, 2, 1)  # bs, state_dims, num_chunk
            # to matach the size of upsampling nn
            x = x.unsqueeze(-1)
            x = layer(x)
            x = x.squeeze(-1)
            x = self.convs[i](x)
            x = x.permute(0, 2, 1)

        assert self.num_output_layers >= 1
        x = self.output_layers(
            x
        )  # input x: bs, 64, 64; output x: bs, 64, out_dim
        x = x[
            :, :, :, None
        ]  # to match the permute operation in SEMActionDeocer forward_layers.
        return x


class HoloBrainValueDecoder(HoloBrainActionDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_embed = nn.Identity()

    def forward(
        self,
        feature_maps,
        inputs,
        action=None,
        mobile_traj=None,
        text_dict=None,
        num_parallel=None,
        need_apply_scale_shift=False,
        need_recompute=False,
        **kwargs,
    ):
        img_feature = self.format_img_feature_maps(feature_maps)

        if "hist_robot_state" not in inputs:
            hist_robot_state = self.joint_state_to_robot_state(
                inputs["hist_joint_state"], inputs
            )
        else:
            hist_robot_state = inputs["hist_robot_state"]

        joint_scale_shift = inputs.get("joint_scale_shift")
        hist_robot_state = apply_scale_shift(
            hist_robot_state, joint_scale_shift
        )
        bs, hist_steps, num_joint, state_dims = hist_robot_state.shape

        if "joint_relative_pos" in inputs:
            joint_relative_pos = inputs["joint_relative_pos"]
        else:
            joint_relative_pos = torch.stack(
                [k.joint_relative_pos for k in inputs["kinematics"]]
            )
        joint_relative_pos = joint_relative_pos.to(hist_robot_state)

        if self.robot_encoder is not None:
            robot_feature = self.robot_encoder(
                hist_robot_state, joint_relative_pos
            )
        else:
            robot_feature = None

        if action is None:
            action = inputs["pred_robot_state"]
            if self.with_mobile:
                mobile_traj = inputs.get("mobile_traj")

        if action.dim() == 5 or num_parallel is not None:
            if num_parallel is None:
                num_parallel = action.shape[1]
                action = action.flatten(0, 1)
                if mobile_traj is not None:
                    mobile_traj = mobile_traj.flatten(0, 1)
            text_dict = text_dict.copy()
            (
                img_feature,
                robot_feature,
                joint_relative_pos,
                text_dict["embedded"],
                text_dict["text_token_mask"],
            ) = self._repeat(
                num_parallel,
                img_feature,
                robot_feature,
                joint_relative_pos,
                text_dict["embedded"],
                text_dict["text_token_mask"],
            )
        else:
            num_parallel = None

        if need_apply_scale_shift:
            action = apply_scale_shift(action, joint_scale_shift)
        if need_recompute:
            action = recompute(action, inputs)

        pred, _ = self.forward_layers(
            action,
            img_feature,
            text_dict,
            robot_feature,
            timesteps=None,
            joint_relative_pos=joint_relative_pos,
            noisy_mobile_traj=mobile_traj,
        )  # call self.head() inside

        # convert back to shape (bs, pred_steps)
        pred = pred.permute(0, 2, 1, 3).squeeze(-1)  # bs, pred_steps, out_dim

        if num_parallel is not None:
            pred = pred.unflatten(0, (bs, num_parallel))

        return pred
