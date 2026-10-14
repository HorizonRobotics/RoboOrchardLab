# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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


import copy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from robo_orchard_core.utils.config import load_config_class

from robo_orchard_lab.models.holobrain_0_5.resources import export_urdf_files
from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.utils.build import DelayInitDictType, build
from robo_orchard_lab.utils.path import in_cwd

__all__ = [
    "HoloBrainProcessor",
    "HoloBrainProcessorCfg",
    "MultiArmManipulationInput",
    "MultiArmManipulationOutput",
]
TENSOR_TYPE = np.ndarray | torch.Tensor


@dataclass
class MultiArmManipulationInput:
    """Data structure for inputs to a multi-arm manipulation pipeline.

    This class defines the expected inputs for a robotics policy that may use
    various modalities like vision, proprioception, and language instructions.
    """

    image: dict[str, list[TENSOR_TYPE]] | None = None
    """A dictionary mapping camera names to lists of RGB images.     Allows
    for multiple viewpoints.
    """
    depth: dict[str, list[TENSOR_TYPE]] | None = None
    "A dictionary mapping camera names to lists of depth maps."
    intrinsic: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their intrinsic     parameter
    matrices.
    """
    t_world2cam: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their world-to-camera
    transformation matrices (e.g., extrinsic parameters).
    """
    t_base2cam: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their base-to-camera
    transformation matrices.
    """
    t_base2world: TENSOR_TYPE | None = None
    """The transformation from the robot's base frame to the     world
    coordinate frame.
    """
    t_base2ego: TENSOR_TYPE | None = None
    """The transformation from the robot's base frame to an     egocentric
    frame, if applicable.
    """
    history_joint_state: list[TENSOR_TYPE] | None = None
    """A list of past joint states, representing the     robot's
    proprioceptive history.
    """
    history_ee_pose: list[TENSOR_TYPE] | None = None
    "A list of past end-effector poses."
    instruction: str | None = None
    "A natural language command or goal for the task."
    urdf: str | None = None
    """The URDF (Unified Robot Description Format) of the robot as a
    string, describing its kinematic and dynamic properties.
    """


@dataclass
class MultiArmManipulationOutput:
    """Data structure for outputs from a multi-arm manipulation pipeline.

    This class encapsulates the results produced by the inference pipeline,
    primarily the predicted action for the robot.
    """

    action: TENSOR_TYPE
    """The predicted action tensor. This could represent target joint
    positions, end-effector velocities, or another action space format.
    """
    pose: TENSOR_TYPE
    "The predicted joint angle position and 6d pose of each joint"


class Struct2Dict:
    """Convert one structured observation into ordered action sample arrays."""

    def __init__(
        self,
        load_image: bool,
        load_depth: bool,
        cam_names: Optional[List[str]] = None,
    ):
        self.load_image = load_image
        self.load_depth = load_depth
        self.cam_names = cam_names

    def __call__(self, data: MultiArmManipulationInput) -> dict:
        input_data = dict()
        if self.cam_names is None:
            cam_names = list(data.intrinsic.keys())
        else:
            cam_names = self.cam_names
        if data.intrinsic is not None:
            input_data["intrinsic"] = np.stack(
                [data.intrinsic[x] for x in cam_names]
            )
        if data.t_world2cam is not None:
            input_data["T_world2cam"] = np.stack(
                [data.t_world2cam[x] for x in cam_names]
            )
        if data.t_base2cam is not None:
            input_data["T_base2cam"] = np.stack(
                [data.t_base2cam[x] for x in cam_names]
            )
        if data.t_base2world is not None:
            input_data["T_base2world"] = data.t_base2world
        if data.t_base2ego is not None:
            input_data["T_base2ego"] = data.t_base2ego
        assert data.history_joint_state is not None
        input_data["joint_state"] = np.stack(data.history_joint_state)
        input_data["step_index"] = len(data.history_joint_state) - 1
        if self.load_image:
            assert data.image is not None
            input_data["imgs"] = [data.image[x][-1] for x in cam_names]
        if self.load_depth:
            assert data.depth is not None
            input_data["depths"] = [data.depth[x][-1] for x in cam_names]
        input_data["text"] = (
            "" if data.instruction is None else data.instruction
        )
        input_data["cam_names"] = cam_names
        return input_data


class HoloBrainProcessor(ModelIOProcessor):
    """Prepare RGB-D and history inputs and convert model trajectories.

    pre_process returns an already batched model input of size one.
    Pass this same dictionary to post_process with model outputs to retain
    kinematic context. Joint order must match the configured embodiment.
    """

    cfg: "HoloBrainProcessorCfg"

    def __init__(self, cfg: "HoloBrainProcessorCfg"):
        super().__init__(cfg)
        self.struction_to_dict = Struct2Dict(
            load_image=self.cfg.load_image,
            load_depth=self.cfg.load_depth,
            cam_names=self.cfg.cam_names,
        )
        self.transforms = (
            [build(transform) for transform in self.cfg.transforms]
            if self.cfg.transforms is not None
            else []
        )

    def pre_process(self, data: Union[MultiArmManipulationInput, Dict]):
        if isinstance(data, MultiArmManipulationInput):
            data = self.struction_to_dict(data)
        for ts_i in self.transforms:
            data = ts_i(data)
        return data

    def post_process(self, model_outputs, batch) -> MultiArmManipulationOutput:
        action = model_outputs[0]["pred_actions"][0][..., 0]
        pose = model_outputs[0]["pred_actions"][0]
        if self.cfg.valid_action_step is not None:
            action = action[: self.cfg.valid_action_step]
            pose = pose[: self.cfg.valid_action_step]
        return MultiArmManipulationOutput(action=action, pose=pose)

    def save(self, path, processor_name, urdf_dir="./urdf"):
        os.makedirs(path, exist_ok=True)
        cfg = copy.deepcopy(self.cfg)
        export_urdf_files(cfg.transforms, path, urdf_dir)
        with open(os.path.join(path, processor_name), "w") as fh:
            fh.write(cfg.model_dump_json(indent=4))

    @staticmethod
    def load(path, processor_name):
        with open(os.path.join(path, processor_name)) as stream:
            processor_cfg = load_config_class(stream.read())
        with in_cwd(path):
            processor = processor_cfg()
        return processor


class HoloBrainProcessorCfg(ModelIOProcessorCfg[HoloBrainProcessor]):
    """Configure camera order, action transforms and output horizon."""

    class_type: ClassType_co[HoloBrainProcessor] = HoloBrainProcessor
    load_image: bool = True
    load_depth: bool = True
    cam_names: Optional[List[str]] = None
    joint_names: Optional[List[str]] = None
    "Model input/output joint order, including virtual gripper slots."
    valid_action_step: Optional[int] = None
    transforms: Optional[List[DelayInitDictType]] = None
