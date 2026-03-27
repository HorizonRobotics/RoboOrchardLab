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

import pytest
import torch
from torch import nn

from robo_orchard_lab.models.holobrain import (
    HoloBrainEncoderBaseConfig,
    HoloBrainEncoderTransformerConfig,
    HoloBrainRobotStateEncoder,
    JointGraphAttention,
    RotaryAttention,
)
from robo_orchard_lab.models.layers.transformer_layers import FFN

EMBED_DIMS = 64
NUM_JOINTS = 7
STATE_DIMS = 8


def _make_encoder(
    chunk_size: int = 1,
    with_temp_attn: bool = False,
) -> HoloBrainRobotStateEncoder:
    """Build a minimal HoloBrainRobotStateEncoder for testing."""
    op_order: list[str | None] = ["norm", "joint_self_attn", "norm", "ffn"]
    temp_self_attn = None
    if with_temp_attn:
        op_order = [
            "norm",
            "joint_self_attn",
            "norm",
            "temp_cross_attn",
            "norm",
            "ffn",
        ]
        temp_self_attn = dict(
            type=RotaryAttention,
            embed_dims=EMBED_DIMS,
            num_heads=4,
            max_position_embeddings=32,
        )

    transformer_cfg = HoloBrainEncoderTransformerConfig(
        joint_self_attn=dict(
            type=JointGraphAttention,
            embed_dims=EMBED_DIMS,
            num_heads=4,
        ),
        norm_layer=dict(
            type=nn.RMSNorm,
            normalized_shape=EMBED_DIMS,
        ),
        ffn=dict(
            type=FFN,
            embed_dims=EMBED_DIMS,
            feedforward_channels=128,
            act_cfg=dict(type=nn.SiLU, inplace=True),
        ),
        temp_self_attn=temp_self_attn,
        operation_order=op_order,
    )
    base_cfg = HoloBrainEncoderBaseConfig(
        embed_dims=EMBED_DIMS,
        state_dims=STATE_DIMS,
        chunk_size=chunk_size,
        act_cfg=dict(type=nn.SiLU, inplace=True),
    )
    return HoloBrainRobotStateEncoder(transformer_cfg, base_cfg)


class TestHoloBrainRobotStateEncoderShape:
    """Output shape tests for HoloBrainRobotStateEncoder."""

    @pytest.fixture
    def encoder(self):
        return _make_encoder(chunk_size=1)

    def test_forward_output_shape(self, encoder):
        """Forward returns [bs, num_joints, num_chunks, embed_dims]."""
        bs, num_steps, num_joints = 2, 4, NUM_JOINTS
        robot_state = torch.randn(bs, num_steps, num_joints, STATE_DIMS)
        joint_dist = torch.rand(bs, num_joints, num_joints)

        out = encoder(robot_state, joint_distance=joint_dist)

        num_chunks = num_steps // encoder.chunk_size
        assert out.shape == (bs, num_joints, num_chunks, EMBED_DIMS)

    def test_output_shape_with_chunk_size(self):
        """num_chunks = num_steps // chunk_size."""
        chunk_size = 2
        encoder = _make_encoder(chunk_size=chunk_size)
        bs, num_steps, num_joints = 2, 8, NUM_JOINTS
        robot_state = torch.randn(bs, num_steps, num_joints, STATE_DIMS)
        joint_dist = torch.rand(bs, num_joints, num_joints)

        out = encoder(robot_state, joint_distance=joint_dist)

        assert out.shape == (
            bs,
            num_joints,
            num_steps // chunk_size,
            EMBED_DIMS,
        )

    def test_single_batch(self, encoder):
        """Works with batch size 1."""
        robot_state = torch.randn(1, 4, NUM_JOINTS, STATE_DIMS)
        joint_dist = torch.rand(1, NUM_JOINTS, NUM_JOINTS)
        out = encoder(robot_state, joint_distance=joint_dist)
        assert out.shape == (1, NUM_JOINTS, 4, EMBED_DIMS)


class TestHoloBrainRobotStateEncoderMask:
    """Joint mask tests for HoloBrainRobotStateEncoder."""

    @pytest.fixture
    def encoder(self):
        return _make_encoder(chunk_size=1)

    def test_forward_with_joint_mask(self, encoder):
        """Joint mask is applied without error."""
        bs, num_steps, num_joints = 2, 4, NUM_JOINTS
        robot_state = torch.randn(bs, num_steps, num_joints, STATE_DIMS)
        joint_dist = torch.rand(bs, num_joints, num_joints)
        joint_mask = torch.zeros(bs, num_joints, dtype=torch.bool)
        joint_mask[:, 0] = True  # mask first joint

        out = encoder(
            robot_state, joint_distance=joint_dist, joint_mask=joint_mask
        )

        assert out.shape == (bs, num_joints, num_steps, EMBED_DIMS)

    def test_output_differs_with_and_without_mask(self, encoder):
        """Applying joint mask changes the output."""
        encoder.eval()
        bs, num_steps, num_joints = 2, 4, NUM_JOINTS
        robot_state = torch.randn(bs, num_steps, num_joints, STATE_DIMS)
        joint_dist = torch.rand(bs, num_joints, num_joints)
        joint_mask = torch.zeros(bs, num_joints, dtype=torch.bool)
        joint_mask[:, 0] = True

        with torch.no_grad():
            out_no_mask = encoder(robot_state, joint_distance=joint_dist)
            out_with_mask = encoder(
                robot_state, joint_distance=joint_dist, joint_mask=joint_mask
            )

        assert not torch.allclose(out_no_mask, out_with_mask)


class TestHoloBrainRobotStateEncoderTempAttn:
    """Tests for encoder with temporal self-attention."""

    def test_forward_with_temp_attn(self):
        """Forward pass works when temporal self-attention is included."""
        encoder = _make_encoder(chunk_size=1, with_temp_attn=True)
        bs, num_steps, num_joints = 2, 4, NUM_JOINTS
        robot_state = torch.randn(bs, num_steps, num_joints, STATE_DIMS)
        joint_dist = torch.rand(bs, num_joints, num_joints)

        out = encoder(robot_state, joint_distance=joint_dist)
        assert out.shape == (bs, num_joints, num_steps, EMBED_DIMS)
