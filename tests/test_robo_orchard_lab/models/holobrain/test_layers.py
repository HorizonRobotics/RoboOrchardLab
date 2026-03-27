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

from robo_orchard_lab.models.holobrain.layers import (
    AdaRMSNorm,
    JointGraphAttention,
    RotaryAttention,
    RotaryEmbedding,
    ScalarEmbedder,
    TemporalJointGraphAttention,
    UpsampleHead,
    linear_act_ln,
)


class TestLinearActLn:
    """Tests for linear_act_ln helper function."""

    def test_basic_structure(self):
        """Returns a list of layers: Linear, Act, LayerNorm blocks."""
        layers = linear_act_ln(embed_dims=64, in_loops=2, out_loops=2)
        assert isinstance(layers, list)
        assert len(layers) > 0
        # Should end with LayerNorm repeated out_loops times
        ln_count = sum(1 for lay in layers if isinstance(lay, nn.LayerNorm))
        assert ln_count == 2

    def test_custom_input_dims(self):
        """First Linear layer uses input_dims when provided."""
        layers = linear_act_ln(
            embed_dims=64, in_loops=1, out_loops=1, input_dims=32
        )
        first_linear = layers[0]
        assert isinstance(first_linear, nn.Linear)
        assert first_linear.in_features == 32
        assert first_linear.out_features == 64

    def test_custom_act_cfg(self):
        """Uses the provided activation config."""
        layers = linear_act_ln(
            embed_dims=64,
            in_loops=1,
            out_loops=1,
            act_cfg=dict(type=nn.SiLU, inplace=True),
        )
        act_layers = [lay for lay in layers if isinstance(lay, nn.SiLU)]
        assert len(act_layers) >= 1


class TestScalarEmbedder:
    """Tests for ScalarEmbedder Fourier feature embedder."""

    def test_output_shape(self):
        """Embeds a batch of scalars to (bs, hidden_size)."""
        embedder = ScalarEmbedder(hidden_size=64, frequency_embedding_size=128)
        t = torch.rand(8) * 1000
        out = embedder(t)
        assert out.shape == (8, 64)

    def test_single_scalar(self):
        """Works on a single-element batch."""
        embedder = ScalarEmbedder(hidden_size=32)
        t = torch.tensor([500.0])
        out = embedder(t)
        assert out.shape == (1, 32)

    def test_deterministic_output(self):
        """Same input produces same output (no stochasticity)."""
        embedder = ScalarEmbedder(hidden_size=64)
        embedder.eval()
        t = torch.rand(4) * 1000
        out1 = embedder(t)
        out2 = embedder(t)
        assert torch.allclose(out1, out2)


class TestRotaryEmbedding:
    """Tests for RotaryEmbedding positional encoding."""

    def test_output_shape_matches_input(self):
        """Rotary encoding preserves the shape of the input tensor."""
        rope = RotaryEmbedding(dim=32, max_position_embeddings=64)
        # x: [bs, num_heads, seq_len, head_dim]
        x = torch.randn(2, 4, 16, 32)
        pos = torch.arange(16).unsqueeze(0).expand(2, -1)
        out = rope(x, pos)
        assert out.shape == x.shape

    def test_different_positions_give_different_output(self):
        """Different position IDs produce different encodings."""
        rope = RotaryEmbedding(dim=32, max_position_embeddings=64)
        x = torch.ones(1, 1, 8, 32)
        pos1 = torch.zeros(1, 8, dtype=torch.long)
        pos2 = torch.arange(8).unsqueeze(0)
        out1 = rope(x, pos1)
        out2 = rope(x, pos2)
        assert not torch.allclose(out1, out2)


class TestRotaryAttention:
    """Tests for RotaryAttention multi-head attention with RoPE."""

    @pytest.fixture
    def attn(self):
        return RotaryAttention(
            embed_dims=64, num_heads=4, max_position_embeddings=32
        )

    def test_forward_basic_shape(self, attn):
        """Output has same shape as query."""
        bs, n, c = 2, 16, 64
        query = torch.randn(bs, n, c)
        key = torch.randn(bs, n, c)
        out = attn(query, key)
        assert out.shape == (bs, n, c)

    def test_forward_with_position_ids(self, attn):
        """Forward pass with positional encodings does not raise."""
        bs, n, m, c = 2, 8, 12, 64
        query = torch.randn(bs, n, c)
        key = torch.randn(bs, m, c)
        query_pos = torch.arange(n).unsqueeze(0).expand(bs, -1)
        key_pos = torch.arange(m).unsqueeze(0).expand(bs, -1)
        out = attn(query, key, query_pos=query_pos, key_pos=key_pos)
        assert out.shape == (bs, n, c)

    def test_identity_residual(self, attn):
        """Uses the provided identity tensor as residual."""
        bs, n, c = 2, 8, 64
        query = torch.randn(bs, n, c)
        key = torch.randn(bs, n, c)
        identity = torch.zeros(bs, n, c)
        with torch.no_grad():
            out = attn(query, key, identity=identity)
        # output = proj(attn(q,k,v)) + identity; identity=0 so just proj output
        assert out.shape == (bs, n, c)

    def test_cross_attention_different_seq_len(self, attn):
        """Cross-attention between query and key of different lengths."""
        bs, n_q, n_k, c = 2, 6, 14, 64
        query = torch.randn(bs, n_q, c)
        key = torch.randn(bs, n_k, c)
        out = attn(query, key)
        assert out.shape == (bs, n_q, c)


class TestJointGraphAttention:
    """Tests for JointGraphAttention over robot joints."""

    @pytest.fixture
    def attn(self):
        return JointGraphAttention(embed_dims=64, num_heads=4)

    def test_forward_basic_shape(self, attn):
        """Output has same (bs, num_joints, embed_dims) shape as query."""
        bs, num_joints, c = 4, 7, 64
        x = torch.randn(bs, num_joints, c)
        # query_pos: [bs_or_1, num_joints, num_joints] joint distance matrix
        joint_dist = torch.rand(1, num_joints, num_joints)
        out = attn(query=x, key=x, query_pos=joint_dist)
        assert out.shape == (bs, num_joints, c)

    def test_identity_residual_used(self, attn):
        """Custom identity tensor is used as the residual."""
        bs, num_joints, c = 2, 7, 64
        x = torch.randn(bs, num_joints, c)
        identity = torch.zeros(bs, num_joints, c)
        joint_dist = torch.rand(1, num_joints, num_joints)
        out = attn(query=x, key=x, query_pos=joint_dist, identity=identity)
        assert out.shape == (bs, num_joints, c)


class TestTemporalJointGraphAttention:
    """Tests for TemporalJointGraphAttention."""

    @pytest.fixture
    def attn(self):
        return TemporalJointGraphAttention(
            embed_dims=64, num_heads=4, max_position_embeddings=32
        )

    def test_forward_shape(self, attn):
        """Output has same (bs, num_joints, num_chunks, embed_dims) shape."""
        bs, num_joints, num_chunks, c = 2, 7, 4, 64
        query = torch.randn(bs, num_joints, num_chunks, c)
        key = torch.randn(bs, num_joints, num_chunks, c)
        joint_dist = torch.rand(bs, num_joints, num_joints)
        # temporal_pos_q/k: [bs, num_chunks], matching the decoder's usage
        temporal_pos = torch.arange(num_chunks).unsqueeze(0).expand(bs, -1)
        out = attn(
            query=query,
            key=key,
            joint_distance=joint_dist,
            temporal_pos_q=temporal_pos,
            temporal_pos_k=temporal_pos,
        )
        assert out.shape == (bs, num_joints, num_chunks, c)


class TestAdaRMSNorm:
    """Tests for AdaRMSNorm adaptive normalization."""

    @pytest.fixture
    def norm(self):
        return AdaRMSNorm(normalized_shape=64, condition_dims=128, zero=False)

    @pytest.fixture
    def norm_zero(self):
        return AdaRMSNorm(normalized_shape=64, condition_dims=128, zero=True)

    def test_forward_returns_tuple(self, norm):
        """forward() returns a tuple of 5 elements."""
        x = torch.randn(2, 4, 64)
        c = torch.randn(2, 128)
        result = norm(x, c)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_output_x_shape(self, norm):
        """First element of tuple (x) has same shape as input."""
        x = torch.randn(2, 4, 64)
        c = torch.randn(2, 128)
        out_x, *_ = norm(x, c)
        assert out_x.shape == x.shape

    def test_none_gates_when_not_zero(self, norm):
        """gate_msa/shift_mlp/scale_mlp/gate_mlp are None when zero=False."""
        x = torch.randn(2, 4, 64)
        c = torch.randn(2, 128)
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(x, c)
        assert gate_msa is None
        assert shift_mlp is None
        assert scale_mlp is None
        assert gate_mlp is None

    def test_non_none_gates_when_zero(self, norm_zero):
        """gate_msa/shift_mlp/scale_mlp/gate_mlp are tensors when zero=True."""
        x = torch.randn(2, 4, 64)
        c = torch.randn(2, 128)
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm_zero(x, c)
        assert gate_msa is not None
        assert shift_mlp is not None
        assert scale_mlp is not None
        assert gate_mlp is not None

    def test_3d_input_with_condition(self, norm):
        """Handles 3D input (bs, seq_len, dims) with 2D condition."""
        x = torch.randn(2, 8, 64)
        c = torch.randn(2, 128)
        out_x, *_ = norm(x, c)
        assert out_x.shape == x.shape


class TestUpsampleHead:
    """Tests for UpsampleHead upsampling prediction head."""

    @pytest.fixture
    def head(self):
        return UpsampleHead(
            upsample_sizes=[8, 16],
            input_dim=64,
            dims=[32, 16],
            norm=dict(type=nn.RMSNorm, normalized_shape=64),
            act=dict(type=nn.SiLU, inplace=True),
            norm_act_idx=[0, 1],
            num_output_layers=1,
            out_dim=8,
        )

    def test_forward_shape(self, head):
        """Output shape is [bs, num_joints, pred_steps, out_dim]."""
        bs, num_joints, num_chunks, c = 2, 7, 4, 64
        x = torch.randn(bs, num_joints, num_chunks, c)
        out = head(x)
        # Final upsample_size = 16, out_dim = 8
        assert out.shape == (bs, num_joints, 16, 8)

    def test_forward_no_output_layers(self):
        """Without output_layers, output last upsampled dim."""
        head = UpsampleHead(
            upsample_sizes=[8],
            input_dim=64,
            dims=[32],
            norm=dict(type=nn.RMSNorm, normalized_shape=64),
            act=dict(type=nn.SiLU, inplace=True),
            norm_act_idx=[0],
            num_output_layers=0,
            out_dim=8,
        )
        bs, num_joints, num_chunks, c = 2, 7, 4, 64
        x = torch.randn(bs, num_joints, num_chunks, c)
        out = head(x)
        assert out.shape == (bs, num_joints, 8, 32)
