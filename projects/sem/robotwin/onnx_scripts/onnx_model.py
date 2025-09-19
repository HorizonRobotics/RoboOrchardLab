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

import os
import shutil

import onnxruntime as ort
import torch
from misc import (
    DECODER_OUTPUT_NAMES,
    ENCODER_INPUT_NAMES,
    ENCODER_OUTPUT_NAMES,
    get_pixels,
)

from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
    build,
)


class OnnxSEMModel(ModelMixin):
    def __init__(self, cfg):
        super().__init__(cfg)
        onnx_path = self.cfg.onnx_path
        if onnx_path is None or not os.path.exists(onnx_path):
            onnx_path = os.environ.get("ORCHARD_LAB_CHECKPOINT_DIRECTORY")
        self.encoder_onnx = os.path.join(onnx_path, "encoder.onnx")
        self.decoder_onnx = os.path.join(onnx_path, "decoder.onnx")
        self.encoder_session = ort.InferenceSession(self.encoder_onnx)
        self.decoder_session = ort.InferenceSession(self.decoder_onnx)
        self.test_noise_scheduler = build(self.cfg.test_noise_scheduler)
        self.data_preprocessor = build(self.cfg.data_preprocessor)
        if isinstance(self.data_preprocessor, torch.nn.Module):
            self.data_preprocessor.eval()
        self.num_inference_timesteps = self.cfg.num_inference_timesteps
        self.strides = self.cfg.strides
        self.pixels = None

    def save_model(self, directory, **kwargs):
        os.makedirs(directory, exist_ok=True)
        config_path = os.path.join(directory, "model.config.json")
        with open(config_path, "w") as f:
            f.write(self.cfg.model_dump_json(indent=4))
        if "encoder.onnx" not in os.listdir(directory):
            shutil.copy(self.encoder_onnx, directory)
        if "decoder.onnx" not in os.listdir(directory):
            shutil.copy(self.decoder_onnx, directory)

    def forward(self, data):
        data["projection_mat_inv"] = torch.linalg.inv(data["projection_mat"])
        if self.data_preprocessor is not None:
            data = self.data_preprocessor(data)

        if self.pixels is None:
            self.pixels = get_pixels(data, self.strides)
        data["pixels"] = self.pixels.clone()

        device = data["imgs"].device
        noisy_action = torch.randn(size=(1, 64, 14, 1), dtype=torch.float32)

        image_feature, robot_feature = self.encoder_session.run(
            ENCODER_OUTPUT_NAMES,
            {key: data[key].cpu().numpy() for key in ENCODER_INPUT_NAMES},
        )
        joint_relative_pos = data["joint_relative_pos"].cpu().numpy()

        self.test_noise_scheduler.set_timesteps(
            self.num_inference_timesteps,
        )
        for t in self.test_noise_scheduler.timesteps:
            noisy_action = self.recompute(noisy_action.to(device), data)

            pred = self.decoder_session.run(
                DECODER_OUTPUT_NAMES,
                {
                    "noisy_action": noisy_action.cpu().numpy(),
                    "image_feature": image_feature,
                    "robot_feature": robot_feature,
                    "timestep": t[None].cpu().numpy().astype("float32"),
                    "joint_relative_pos": joint_relative_pos,
                },
            )[0]

            pred = torch.from_numpy(pred[..., :1]).to(device)
            noisy_action = self.test_noise_scheduler.step(
                pred, t, noisy_action[..., :1]
            ).prev_sample
        pred_actions = self.apply_scale_shift(
            noisy_action,
            data["joint_scale_shift"],
            inverse=True,
        )
        return [dict(pred_actions=pred_actions)]

    # copy from robo_orchard_lab/models/sem_modules/action_decoder.py
    def apply_scale_shift(
        self, robot_state, joint_scale_shift=None, inverse=False
    ):
        if joint_scale_shift is None:
            return robot_state
        scale = joint_scale_shift[:, None, :, 0:1]
        shift = joint_scale_shift[:, None, :, 1:2]
        if not inverse:
            robot_state = torch.cat(
                [(robot_state[..., :1] - shift) / scale, robot_state[..., 1:]],
                dim=-1,
            )
        else:
            robot_state = torch.cat(
                [robot_state[..., :1] * scale + shift, robot_state[..., 1:]],
                dim=-1,
            )
        return robot_state

    def forward_kinematics(self, joint_state, inputs):
        if joint_state.shape[-1] == 1:
            joint_state = joint_state.squeeze(-1)
        robot_state = []
        kinematics = inputs["kinematics"]
        embodiedment_mat = inputs.get(
            "embodiedment_mat", [None] * len(kinematics)
        )
        if len(kinematics) <= 1 or (
            all(x == kinematics[0] for x in kinematics[1:])
            and (
                embodiedment_mat[0] is None
                or (embodiedment_mat[0] - embodiedment_mat[1:] == 0).all()
            )
        ):
            robot_state = kinematics[0].joint_state_to_robot_state(
                joint_state, embodiedment_mat[0]
            )
        else:
            for i in range(len(joint_state)):
                robot_state.append(
                    inputs["kinematics"][i].joint_state_to_robot_state(
                        joint_state[i], embodiedment_mat[i]
                    )
                )
            robot_state = torch.stack(robot_state)
        return robot_state

    def recompute(self, robot_state, inputs):
        joint_state = self.apply_scale_shift(
            robot_state[..., :1],
            inputs.get("joint_scale_shift"),
            inverse=True,
        )
        robot_state = torch.cat(
            [
                robot_state[..., :1],
                self.forward_kinematics(joint_state, inputs)[..., 1:],
            ],
            dim=-1,
        )
        return robot_state


MODULE_TPYE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class OnnxSEMModelConfig(TorchModuleCfg[OnnxSEMModel]):
    class_type: ClassType_co[OnnxSEMModel] = OnnxSEMModel
    onnx_path: str | None = None
    test_noise_scheduler: MODULE_TPYE
    data_preprocessor: MODULE_TPYE | None = None
    num_inference_timesteps: int = 10
    strides: tuple = (16, 32)
