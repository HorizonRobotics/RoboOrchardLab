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
from pytorch3d.transforms import matrix_to_quaternion

from robo_orchard_lab.dataset.egodex.hand_frame import derive_hand_frames


class SimpleStateSampling:
    """Slice history / prediction windows around ``step_index``.

    Consumes ``data["joint_transforms"]`` (T, J, 4, 4) and produces:
        * ``hist_joint_transforms`` — ``hist_steps`` frames ending at
          ``step_index`` inclusive, front-padded with frame 0 if the episode
          starts too close to ``step_index``.
        * ``pred_joint_transforms`` — ``pred_steps`` frames strictly after
          ``step_index``, back-padded with the final frame if the episode ends
          first.
        * ``pred_mask`` — ``(pred_steps,)`` float array; ``1`` at real frames
          and ``0`` at padded slots so downstream losses can ignore padding.
    """

    def __init__(
        self,
        hist_steps,
        pred_steps,
    ):
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps

    def __call__(self, data):
        state = data.pop("joint_transforms")  # T, J, 4, 4
        step_index = data["step_index"]
        hist_steps = self.hist_steps
        pred_steps = self.pred_steps

        # ---- Future window: strictly after step_index --------------------
        pred_state = state[step_index + 1 : step_index + 1 + pred_steps]
        pred_mask = np.zeros(pred_steps)
        pred_mask[: pred_state.shape[0]] = 1

        # Repeat the final frame if the episode ended before we reached
        # pred_steps; mask above already zeroed those slots so the loss
        # ignores the repeated content.
        if pred_state.shape[0] != pred_steps:
            padding = np.tile(
                state[-1:], (pred_steps - pred_state.shape[0], 1, 1, 1)
            )
            pred_state = np.concatenate([pred_state, padding], axis=0)

        # ---- History window: up to and including step_index -------------
        hist_state = state[
            max(0, step_index + 1 - hist_steps) : step_index + 1
        ]
        # Front-pad with the very first frame when the episode starts too
        # close to step_index. No mask here — training treats history as fully
        # observed context.
        if hist_state.shape[0] != hist_steps:
            padding = np.tile(
                state[:1], (hist_steps - hist_state.shape[0], 1, 1, 1)
            )
            hist_state = np.concatenate([padding, hist_state], axis=0)

        data.update(
            hist_joint_transforms=hist_state,
            pred_joint_transforms=pred_state,
            pred_mask=pred_mask,
        )
        return data


class HandTF2Gripper:
    """Convert per-joint hand transforms into two-hand 8D gripper state.

    Runs :func:`derive_hand_frames` on any of ``joint_transforms``,
    ``hist_joint_transforms``, and ``pred_joint_transforms`` present in the
    sample, and writes the result to the matching ``*_robot_state`` key. Each
    hand contributes an 8D vector: ``[openness, xyz, quaternion_wxyz]``, with
    left before right along the added second-to-last axis.
    """

    def __call__(self, data):
        joint_names = data["joint_names"]
        if "hist_joint_transforms" in data:
            data["hist_robot_state"] = self.tf2gripper(
                data["hist_joint_transforms"], joint_names
            )
        if "pred_joint_transforms" in data:
            data["pred_robot_state"] = self.tf2gripper(
                data["pred_joint_transforms"], joint_names
            )
        if "joint_transforms" in data:
            data["robot_state"] = self.tf2gripper(
                data["joint_transforms"], joint_names
            )
        return data

    def tf2gripper(
        self, tf: np.ndarray | torch.Tensor, joint_names: list[str]
    ) -> torch.Tensor:
        """Convert tracked hand transforms into two 8D gripper states.

        Args:
            tf (np.ndarray | torch.Tensor): Joint transforms shaped
                ``[..., J, 4, 4]``. NumPy inputs are copied to a CPU tensor
                while preserving their dtype; tensor inputs preserve both
                dtype and device.
            joint_names (list[str]): Names corresponding to the joint axis.

        Returns:
            torch.Tensor: States shaped ``[..., 2, 8]`` ordered as left then
                right, with ``[openness, xyz, quaternion]`` per hand.
        """
        if not isinstance(tf, torch.Tensor):
            tf = torch.tensor(tf)
        hand_frames, _ = derive_hand_frames(tf, joint_names)
        states = []
        for side in ("left", "right"):
            frame = hand_frames[side]
            states.append(
                torch.cat(
                    [
                        frame.openness,
                        frame.origin,
                        matrix_to_quaternion(frame.rotation),
                    ],
                    dim=-1,
                )
            )
        return torch.stack(states, dim=-2)


class UpSampleRobotState:
    """Resample robot-state windows to a fixed ``pred_steps`` (and history).

    The prediction window is up-sampled together with its mask by prepending
    the last history frame as an anchor, running 1D linear interpolation, and
    dropping the anchor afterwards. Doing so gives the interpolation a real
    start point so the first predicted step is not extrapolated from thin air.

    ``hist_steps`` is optional; when set and different from the current
    history length, the history is separately resampled to that length.
    """

    def __init__(self, pred_steps, hist_steps=None):
        self.pred_steps = pred_steps
        self.hist_steps = hist_steps

    def __call__(self, data):
        # Anchor the interpolation at the last observed history frame so the
        # first predicted step is a genuine interpolation, not an extrapolation
        # from an empty prefix.
        robot_state = torch.cat(
            [data["hist_robot_state"][-1:], data["pred_robot_state"]]
        )  # steps x num_joint x 8
        state_dim = robot_state.shape[-1]
        pred_mask = torch.cat([data["pred_mask"][:1], data["pred_mask"]])[
            :, None
        ]
        # Fold the per-hand state and the mask into a single channel dim so a
        # single interpolate() call handles both consistently.
        robot_state = torch.cat(
            [robot_state.flatten(-2), pred_mask.to(robot_state)], dim=-1
        )
        robot_state = robot_state.T[None]  # 1 x [num_joint*8] x steps

        # Linear interpolation on states, including rotation quaternions.
        robot_state = torch.nn.functional.interpolate(
            robot_state, self.pred_steps + 1, mode="linear", align_corners=True
        )
        # Drop the anchor frame (index 0) and the trailing mask channel. The
        # remaining tensor is reshaped back to (pred_steps, num_hand, 8).
        data["pred_robot_state"] = (
            robot_state[0].T[1:, :-1].unflatten(-1, (-1, state_dim))
        )
        data["pred_mask"] = robot_state[0].T[1:, -1].to(dtype=torch.bool)
        if (
            self.hist_steps is not None
            and data["hist_robot_state"].shape[0] != self.hist_steps
        ):
            # History has no mask and no anchor — resample directly.
            data["hist_robot_state"] = torch.nn.functional.interpolate(
                data["hist_robot_state"].flatten(-2).T[None],
                self.hist_steps,
                mode="linear",
                align_corners=True,
            )[0].T.unflatten(-1, (-1, state_dim))
        return data
