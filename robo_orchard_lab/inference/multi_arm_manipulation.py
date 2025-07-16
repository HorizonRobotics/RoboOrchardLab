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
from dataclasses import dataclass

import numpy as np
import torch

from robo_orchard_lab.inference.basic import (
    ClassType_co,
    InferencePipeline,
    InferencePipelineCfg,
)

__all__ = [
    "MultiArmManipulationInput",
    "MultiArmManipulationOutput",
    "MultiArmManipulationPipeline",
    "MultiArmManipulationPipelineCfg",
]


TENSOR_TYPE = np.ndarray | torch.Tensor


@dataclass
class MultiArmManipulationInput:
    """Data structure for inputs to a multi-arm manipulation pipeline.

    This class defines the expected inputs for a robotics policy that may use
    various modalities like vision, proprioception, and language instructions.
    """

    image: dict[str, list[TENSOR_TYPE]] | None = None
    """A dictionary mapping camera names to lists of RGB images.
    Allows for multiple viewpoints."""

    depth: dict[str, list[TENSOR_TYPE]] | None = None
    """A dictionary mapping camera names to lists of depth maps."""

    intrinsic: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their intrinsic
    parameter matrices."""

    t_world2cam: dict[str, TENSOR_TYPE] | None = None
    """A dictionary mapping camera names to their world-to-camera
    transformation matrices (e.g., extrinsic parameters)."""

    t_robot2world: TENSOR_TYPE | None = None
    """The transformation from the robot's base frame to the
    world coordinate frame."""

    t_robot2ego: TENSOR_TYPE | None = None
    """The transformation from the robot's base frame to an
    egocentric frame, if applicable."""

    history_joint_state: list[TENSOR_TYPE] | None = None
    """A list of past joint states, representing the
    robot's proprioceptive history."""

    history_ee_pose: list[TENSOR_TYPE] | None = None
    """A list of past end-effector poses."""

    instruction: str | None = None
    """A natural language command or goal for the task."""

    urdf: str | None = None
    """The URDF (Unified Robot Description Format) of the robot as a
    string, describing its kinematic and dynamic properties."""


@dataclass
class MultiArmManipulationOutput:
    """Data structure for outputs from a multi-arm manipulation pipeline.

    This class encapsulates the results produced by the inference pipeline,
    primarily the predicted action for the robot.
    """

    action: TENSOR_TYPE
    """The predicted action tensor. This could represent target joint
    positions, end-effector velocities, or another action space format."""


class MultiArmManipulationPipeline(
    InferencePipeline[MultiArmManipulationInput, MultiArmManipulationOutput]
):
    """An inference pipeline specialized for multi-arm manipulation tasks.

    This class specializes the generic `InferencePipeline` for the specific
    needs of multi-arm manipulation by defining the concrete input and output
    data structures. It inherits the standard inference workflow (pre-process,
    collate, forward, post-process) from its parent class.
    """

    cfg: "MultiArmManipulationPipelineCfg"  # for type hint


class MultiArmManipulationPipelineCfg(
    InferencePipelineCfg[MultiArmManipulationPipeline]
):
    """Configuration class for the `MultiArmManipulationPipeline`.

    This class links the `MultiArmManipulationPipeline` to the configuration
    system, allowing it to be instantiated dynamically from a configuration
    file. It sets the `class_type` to point directly to the
    `MultiArmManipulationPipeline`.
    """

    class_type: ClassType_co[MultiArmManipulationPipeline] = (
        MultiArmManipulationPipeline  # noqa: E501
    )
