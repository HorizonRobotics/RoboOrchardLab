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

from robo_orchard_lab.models.holobrain.utils import (
    apply_joint_mask,
    apply_scale_shift,
    recompute,
)


class TestApplyScaleShift:
    """Tests for apply_scale_shift utility function."""

    def test_identity_when_none(self):
        """Returns the input unchanged when joint_scale_shift is None."""
        robot_state = torch.randn(2, 4, 7, 8)
        result = apply_scale_shift(robot_state, None)
        assert result is robot_state

    def test_forward_normalization(self):
        """Forward pass: normalized = (original - shift) / scale."""
        bs, num_steps, num_joints, c = 2, 4, 7, 8
        robot_state = torch.randn(bs, num_steps, num_joints, c)
        # scale_shift[:, :, 0] = scale, scale_shift[:, :, 1] = shift
        scale = torch.full((bs, num_joints, 1), 2.0)
        shift = torch.full((bs, num_joints, 1), 1.0)
        joint_scale_shift = torch.cat([scale, shift], dim=-1)

        result = apply_scale_shift(
            robot_state, joint_scale_shift, inverse=False
        )

        expected_angle = (robot_state[..., :1] - 1.0) / 2.0
        assert torch.allclose(result[..., :1], expected_angle, atol=1e-5)
        # Other channels should remain unchanged
        assert torch.allclose(result[..., 1:], robot_state[..., 1:], atol=1e-5)

    def test_inverse_normalization(self):
        """Inverse pass: original = normalized * scale + shift."""
        bs, num_steps, num_joints, c = 2, 4, 7, 8
        robot_state = torch.randn(bs, num_steps, num_joints, c)
        scale = torch.full((bs, num_joints, 1), 2.0)
        shift = torch.full((bs, num_joints, 1), 1.0)
        joint_scale_shift = torch.cat([scale, shift], dim=-1)

        result = apply_scale_shift(
            robot_state, joint_scale_shift, inverse=True
        )

        expected_angle = robot_state[..., :1] * 2.0 + 1.0
        assert torch.allclose(result[..., :1], expected_angle, atol=1e-5)
        assert torch.allclose(result[..., 1:], robot_state[..., 1:], atol=1e-5)

    def test_scale_only_ignores_shift(self):
        """When scale_only=True, shift is treated as zero."""
        bs, num_steps, num_joints, c = 2, 4, 7, 8
        robot_state = torch.randn(bs, num_steps, num_joints, c)
        scale = torch.full((bs, num_joints, 1), 2.0)
        shift = torch.full((bs, num_joints, 1), 999.0)  # should be ignored
        joint_scale_shift = torch.cat([scale, shift], dim=-1)

        result = apply_scale_shift(
            robot_state, joint_scale_shift, inverse=False, scale_only=True
        )

        expected_angle = robot_state[..., :1] / 2.0
        assert torch.allclose(result[..., :1], expected_angle, atol=1e-5)

    def test_forward_inverse_roundtrip(self):
        """Forward followed by inverse returns the original tensor."""
        bs, num_steps, num_joints, c = 2, 4, 7, 8
        robot_state = torch.randn(bs, num_steps, num_joints, c)
        scale = torch.rand(bs, num_joints, 1) + 0.5
        shift = torch.randn(bs, num_joints, 1)
        joint_scale_shift = torch.cat([scale, shift], dim=-1)

        normalized = apply_scale_shift(
            robot_state, joint_scale_shift, inverse=False
        )
        recovered = apply_scale_shift(
            normalized, joint_scale_shift, inverse=True
        )
        assert torch.allclose(recovered, robot_state, atol=1e-5)

    def test_batch_size_mismatch_with_num_parallel(self):
        """Handles batch size mismatch by repeating scale_shift."""
        num_parallel = 3
        bs = 2
        num_steps, num_joints, c = 4, 7, 8
        # robot_state has bs * num_parallel in batch dim
        robot_state = torch.randn(bs * num_parallel, num_steps, num_joints, c)
        scale = torch.full((bs, num_joints, 1), 2.0)
        shift = torch.full((bs, num_joints, 1), 1.0)
        joint_scale_shift = torch.cat([scale, shift], dim=-1)

        # Should not raise, repeat_interleave is applied internally
        result = apply_scale_shift(
            robot_state, joint_scale_shift, inverse=False
        )
        assert result.shape == robot_state.shape


class TestApplyJointMask:
    """Tests for apply_joint_mask utility function."""

    def test_masked_joints_set_to_default_constant(self):
        """Masked joints are set to -1 by default."""
        bs, num_joints, num_steps, c = 2, 7, 4, 8
        robot_state = torch.randn(bs, num_joints, num_steps, c)
        joint_mask = torch.zeros(bs, num_joints, dtype=torch.bool)
        joint_mask[:, 0] = True  # mask first joint

        result = apply_joint_mask(robot_state, joint_mask)

        assert torch.all(result[:, 0, :, :1] == -1)
        assert torch.allclose(
            result[:, 1:, :, :1], robot_state[:, 1:, :, :1], atol=1e-6
        )

    def test_custom_constant_value(self):
        """Masked joints are set to the specified constant value."""
        bs, num_joints, num_steps, c = 2, 7, 4, 8
        robot_state = torch.randn(bs, num_joints, num_steps, c)
        joint_mask = torch.ones(bs, num_joints, dtype=torch.bool)

        result = apply_joint_mask(robot_state, joint_mask, constant_value=0)

        assert torch.all(result[..., :1] == 0.0)

    def test_non_first_channels_unchanged(self):
        """Channels beyond the first (pose channels) are never masked."""
        bs, num_joints, num_steps, c = 2, 7, 4, 8
        robot_state = torch.randn(bs, num_joints, num_steps, c)
        joint_mask = torch.ones(bs, num_joints, dtype=torch.bool)

        result = apply_joint_mask(robot_state, joint_mask)

        assert torch.allclose(result[..., 1:], robot_state[..., 1:], atol=1e-6)

    def test_no_mask_unchanged(self):
        """When mask is all False, robot_state is unchanged."""
        bs, num_joints, num_steps, c = 2, 7, 4, 8
        robot_state = torch.randn(bs, num_joints, num_steps, c)
        joint_mask = torch.zeros(bs, num_joints, dtype=torch.bool)

        result = apply_joint_mask(robot_state, joint_mask)

        assert torch.allclose(result, robot_state, atol=1e-6)

    def test_output_shape_unchanged(self):
        """Output shape matches input shape."""
        bs, num_joints, num_steps, c = 3, 5, 8, 8
        robot_state = torch.randn(bs, num_joints, num_steps, c)
        joint_mask = torch.rand(bs, num_joints) > 0.5

        result = apply_joint_mask(robot_state, joint_mask)

        assert result.shape == robot_state.shape


class TestRecompute:
    """Tests for recompute utility function."""

    def test_passthrough_without_kinematics(self):
        """Returns input unchanged when 'kinematics' is not in inputs."""
        bs, num_steps, num_joints, c = 2, 4, 7, 8
        robot_state = torch.randn(bs, num_steps, num_joints, c)
        inputs = {}

        result = recompute(robot_state, inputs)

        assert result is robot_state
