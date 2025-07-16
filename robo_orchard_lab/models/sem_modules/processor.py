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

import numpy as np
import torch

from robo_orchard_lab.dataset.robotwin.transforms import (
    AddScaleShift,
    ConvertDataType,
    DualArmKinematics,
    GetProjectionMat,
    ItemSelection,
    Resize,
    SimpleStateSampling,
    ToTensor,
)
from robo_orchard_lab.inference.multi_arm_manipulation import (
    MultiArmManipulationInput,
    MultiArmManipulationOutput,
)
from robo_orchard_lab.inference.processor import (
    ClassType_co,
    ProcessorMixin,
    ProcessorMixinCfg,
)

__all__ = ["SEMProcessor", "SEMProcessorCfg"]


class Struct2Dict:
    def __init__(self, load_image: bool, load_depth: bool):
        self.load_image = load_image
        self.load_depth = load_depth

    def __call__(self, data: MultiArmManipulationInput) -> dict:
        input_data = dict()

        assert data.intrinsic is not None
        input_data["intrinsic"] = np.stack(tuple(data.intrinsic.values()))

        assert data.t_world2cam is not None
        input_data["T_world2cam"] = np.stack(tuple(data.t_world2cam.values()))

        assert data.t_robot2world is not None
        input_data["T_base2world"] = data.t_robot2world

        if data.t_robot2ego is not None:
            input_data["T_base2ego"] = data.t_robot2ego

        assert data.history_joint_state is not None
        input_data["joint_state"] = np.stack(data.history_joint_state)
        input_data["step_index"] = len(data.history_joint_state) - 1

        if self.load_image:
            assert data.image is not None
            images = [images[-1] for images in data.image.values()]
            input_data["imgs"] = np.stack(images)

        if self.load_depth:
            assert data.depth is not None
            depth = [depth[-1] for depth in data.depth.values()]
            input_data["depths"] = np.stack(depth)

        input_data["text"] = (
            "" if data.instruction is None else data.instruction
        )

        return input_data


class SEMProcessor(ProcessorMixin):
    cfg: "SEMProcessorCfg"  # for type hint

    def __init__(self, cfg: "SEMProcessorCfg"):
        super().__init__(cfg)
        self.ts = None

    def _initialized(self, urdf: str | None):
        if self.ts is not None:
            return

        if urdf is None:
            raise ValueError("Required urdf!")

        self.ts = [
            Struct2Dict(
                load_image=self.cfg.load_image,
                load_depth=self.cfg.load_depth,
            ),
            SimpleStateSampling(
                hist_steps=self.cfg.hist_steps, pred_steps=self.cfg.pred_steps
            ),
            Resize(
                dst_wh=self.cfg.resize_dst_wh,
                dst_intrinsic=self.cfg.resize_dst_intrinsic,
            ),
            ToTensor(),
            GetProjectionMat(target_coordinate="base"),
            AddScaleShift(scale_shift=self.cfg.scale_shift_list),
            ConvertDataType(
                convert_map=dict(
                    imgs=torch.float32,
                    depths=torch.float32,
                    image_wh=torch.float32,
                    projection_mat=torch.float32,
                    embodiedment_mat=torch.float32,
                )
            ),
            DualArmKinematics(urdf=urdf),
            ItemSelection(
                keys=[
                    "imgs",
                    "depths",
                    "image_wh",
                    "projection_mat",
                    "embodiedment_mat",
                    "hist_robot_state",
                    "pred_robot_state",
                    "joint_relative_pos",
                    "joint_scale_shift",
                    "kinematics",
                    "text",
                    "uuid",
                ]
            ),
        ]

    def pre_process(self, data: MultiArmManipulationInput):
        self._initialized(data.urdf)
        for ts_i in self.ts:  # type: ignore
            data = ts_i(data)
        return data

    def post_process(self, batch, model_outputs) -> MultiArmManipulationOutput:
        action = model_outputs[0]["pred_actions"][0][
            : self.cfg.valid_action_step, :, 0
        ]
        return MultiArmManipulationOutput(action=action)


class SEMProcessorCfg(ProcessorMixinCfg[SEMProcessor]):
    class_type: ClassType_co[SEMProcessor] = SEMProcessor
    load_image: bool = True
    load_depth: bool = True
    hist_steps: int = 1
    pred_steps: int = 64
    resize_dst_wh: tuple[int, int] | list[int]
    resize_dst_intrinsic: list[list[float]]
    scale_shift_list: list[list[float]]
    valid_action_step: int = 32
