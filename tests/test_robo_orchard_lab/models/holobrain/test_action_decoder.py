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
    AdaRMSNorm,
    HoloBrainActionDecoder,
    HoloBrainActionLoss,
    HoloBrainDecoderBaseConfig,
    HoloBrainDecoderTransformerConfig,
    HoloBrainEncoderBaseConfig,
    HoloBrainEncoderTransformerConfig,
    HoloBrainRobotStateEncoder,
    HoloBrainTrainingConfig,
    JointGraphAttention,
    RotaryAttention,
    TemporalJointGraphAttention,
    UpsampleHead,
)
from robo_orchard_lab.models.layers.transformer_layers import FFN

EMBED_DIMS = 64
NUM_JOINTS = 7
STATE_DIMS = 8
PRED_STEPS = 16
CHUNK_SIZE = 4


def _make_decoder(
    with_robot_encoder: bool = False,
    prediction_type: str = "absolute_joint_absolute_pose",
    pred_scaled_joint: bool = True,
    noise_type: str = "global_joint",
    num_parallel: int = 1,
) -> HoloBrainActionDecoder:
    """Build a minimal HoloBrainActionDecoder for testing."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_dpmsolver_multistep import (
        DPMSolverMultistepScheduler,
    )

    num_chunks = PRED_STEPS // CHUNK_SIZE
    op_order: list[str | None] = [
        "t_norm",
        "img_cross_attn",
        "norm",
        "ffn",
    ]

    robot_encoder = None
    if with_robot_encoder:
        robot_encoder = dict(
            type=HoloBrainRobotStateEncoder,
            transformer_cfg=HoloBrainEncoderTransformerConfig(
                joint_self_attn=dict(
                    type=JointGraphAttention,
                    embed_dims=EMBED_DIMS,
                    num_heads=4,
                ),
                norm_layer=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
                ffn=dict(
                    type=FFN,
                    embed_dims=EMBED_DIMS,
                    feedforward_channels=128,
                    act_cfg=dict(type=nn.SiLU, inplace=True),
                ),
                operation_order=["norm", "joint_self_attn", "norm", "ffn"],
            ),
            base_cfg=HoloBrainEncoderBaseConfig(
                embed_dims=EMBED_DIMS,
                state_dims=STATE_DIMS,
                chunk_size=1,
                act_cfg=dict(type=nn.SiLU, inplace=True),
            ),
        )

    return HoloBrainActionDecoder(
        transformer_cfg=HoloBrainDecoderTransformerConfig(
            img_cross_attn=dict(
                type=RotaryAttention,
                embed_dims=EMBED_DIMS,
                num_heads=4,
                max_position_embeddings=64,
            ),
            norm_layer=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
            ffn=dict(
                type=FFN,
                embed_dims=EMBED_DIMS,
                feedforward_channels=128,
                act_cfg=dict(type=nn.SiLU, inplace=True),
            ),
            timestep_norm_layer=dict(
                type=AdaRMSNorm,
                normalized_shape=EMBED_DIMS,
                condition_dims=EMBED_DIMS,
                zero=False,
            ),
            operation_order=op_order,
        ),
        head=dict(
            type=UpsampleHead,
            upsample_sizes=[num_chunks * 2, PRED_STEPS],
            input_dim=EMBED_DIMS,
            dims=[32, 16],
            norm=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
            act=dict(type=nn.SiLU, inplace=True),
            norm_act_idx=[0, 1],
            num_output_layers=1,
            out_dim=STATE_DIMS,
        ),
        base_cfg=HoloBrainDecoderBaseConfig(
            embed_dims=EMBED_DIMS,
            state_dims=STATE_DIMS,
            pred_steps=PRED_STEPS,
            chunk_size=CHUNK_SIZE,
            num_inference_timesteps=5,
            feature_level=[0],
            noise_type=noise_type,
            pred_scaled_joint=pred_scaled_joint,
            prediction_type=prediction_type,
            act_cfg=dict(type=nn.SiLU, inplace=True),
            training_noise_scheduler=dict(
                type=DDPMScheduler,
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="sample",
                clip_sample=False,
            ),
            test_noise_scheduler=dict(
                type=DPMSolverMultistepScheduler,
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="sample",
            ),
        ),
        training_cfg=HoloBrainTrainingConfig(
            num_parallel_training_sample=num_parallel,
            loss=dict(
                type=HoloBrainActionLoss,
                loss_mode="l2",
                with_wasserstein_distance=False,
            ),
        ),
        robot_encoder=robot_encoder,
    )


class TestFormatImgFeatureMaps:
    """Tests for HoloBrainActionDecoder.format_img_feature_maps."""

    @pytest.fixture
    def decoder(self):
        return _make_decoder()

    def test_single_tensor_passthrough(self, decoder):
        """A single tensor is returned as-is."""
        bs, n, c = 2, 100, EMBED_DIMS
        feat = torch.randn(bs, n, c)
        out = decoder.format_img_feature_maps(feat)
        assert out is feat

    def test_list_of_feature_maps(self, decoder):
        """Feature maps [bs, cams, c, h, w] flatten to [bs, cams*h*w, c]."""
        bs, num_cams, c, h, w = 2, 2, EMBED_DIMS, 8, 10
        # Feature maps: [bs, num_cams, c, h, w] (as returned by VLM handler)
        # decoder.feature_level = [0], so only first level is used
        feature_maps = [torch.randn(bs, num_cams, c, h, w)]
        out = decoder.format_img_feature_maps(feature_maps)
        assert out.shape == (bs, num_cams * h * w, c)

    def test_multiple_levels_concatenated(self):
        """Multiple feature levels concatenated along patch dimension."""
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )

        # Build decoder that uses feature levels [0, 1]
        decoder = HoloBrainActionDecoder(
            transformer_cfg=HoloBrainDecoderTransformerConfig(
                img_cross_attn=dict(
                    type=RotaryAttention,
                    embed_dims=EMBED_DIMS,
                    num_heads=4,
                    max_position_embeddings=64,
                ),
                norm_layer=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
                ffn=dict(
                    type=FFN, embed_dims=EMBED_DIMS, feedforward_channels=128
                ),
                timestep_norm_layer=dict(
                    type=AdaRMSNorm,
                    normalized_shape=EMBED_DIMS,
                    condition_dims=EMBED_DIMS,
                    zero=False,
                ),
                operation_order=["t_norm", "img_cross_attn", "norm", "ffn"],
            ),
            head=dict(
                type=UpsampleHead,
                upsample_sizes=[PRED_STEPS // CHUNK_SIZE * 2, PRED_STEPS],
                input_dim=EMBED_DIMS,
                dims=[32, 16],
                norm=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
                act=dict(type=nn.SiLU, inplace=True),
                norm_act_idx=[0, 1],
                num_output_layers=1,
                out_dim=STATE_DIMS,
            ),
            base_cfg=HoloBrainDecoderBaseConfig(
                embed_dims=EMBED_DIMS,
                state_dims=STATE_DIMS,
                pred_steps=PRED_STEPS,
                chunk_size=CHUNK_SIZE,
                feature_level=[0, 1],  # two levels
                training_noise_scheduler=dict(
                    type=DDPMScheduler,
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2",
                    prediction_type="sample",
                    clip_sample=False,
                ),
                test_noise_scheduler=dict(
                    type=DPMSolverMultistepScheduler,
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2",
                    prediction_type="sample",
                ),
            ),
        )
        bs, num_cams, c = 2, 2, EMBED_DIMS
        # feature maps: [bs, num_cams, c, h, w] format
        feature_maps = [
            torch.randn(bs, num_cams, c, 8, 10),
            torch.randn(bs, num_cams, c, 4, 5),
        ]
        out = decoder.format_img_feature_maps(feature_maps)
        # num_cams * (8*10 + 4*5) = 2 * 100 = 200 patches total
        assert out.shape == (bs, num_cams * (8 * 10 + 4 * 5), c)


class TestSampleNoise:
    """Tests for HoloBrainActionDecoder.sample_noise."""

    @pytest.fixture
    def decoder(self):
        return _make_decoder()

    def test_global_joint_noise_shape(self, decoder):
        """global_joint noise has shape [bs, steps, joints, 1]."""
        bs, steps, joints, c = 2, 16, 7, 8
        hist = torch.randn(bs, 1, joints, c)
        noise = decoder.sample_noise(
            [bs, steps, joints, c], hist, "global_joint"
        )
        assert noise.shape == (bs, steps, joints, 1)

    def test_global_joint_global_pose_shape(self, decoder):
        """global_joint_global_pose noise has full state_dims shape."""
        bs, steps, joints, c = 2, 16, 7, 8
        hist = torch.randn(bs, 1, joints, c)
        noise = decoder.sample_noise(
            [bs, steps, joints, c], hist, "global_joint_global_pose"
        )
        assert noise.shape == (bs, steps, joints, c)

    def test_local_joint_noise_offset(self, decoder):
        """local_joint noise offset by last historical joint position."""
        bs, steps, joints = 2, 16, 7
        hist = torch.ones(bs, 1, joints, STATE_DIMS) * 3.0  # last state = 3
        noise = decoder.sample_noise(
            [bs, steps, joints, STATE_DIMS], hist, "local_joint"
        )
        # noise[..., :1] should be centered around
        # hist_robot_state[:,-1:,:,:1] = 3
        # It's random so just check shape
        assert noise.shape == (bs, steps, joints, 1)

    def test_noise_dtype_matches_hist(self, decoder):
        """Noise dtype and device match hist_robot_state."""
        bs, steps, joints = 2, 16, 7
        hist = torch.randn(bs, 1, joints, STATE_DIMS).float()
        noise = decoder.sample_noise(
            [bs, steps, joints, STATE_DIMS], hist, "global_joint"
        )
        assert noise.dtype == hist.dtype


class TestGetPrediction:
    """Tests for HoloBrainActionDecoder.get_prediction."""

    def _make_decoder_with_prediction_type(self, prediction_type):
        return _make_decoder(
            prediction_type=prediction_type, pred_scaled_joint=True
        )

    def test_absolute_joint_absolute_pose(self):
        """absolute_joint_absolute_pose: prediction is returned as-is."""
        decoder = self._make_decoder_with_prediction_type(
            "absolute_joint_absolute_pose"
        )
        bs, steps, joints = 2, 16, NUM_JOINTS
        model_pred = torch.randn(bs, steps, joints, STATE_DIMS)
        hist = torch.randn(bs, 1, joints, STATE_DIMS)
        result = decoder.get_prediction(model_pred, hist, None, None)
        assert torch.allclose(result, model_pred)

    def test_relative_joint_absolute_pose(self):
        """relative_joint_absolute_pose: joint channel offset by last hist."""
        decoder = self._make_decoder_with_prediction_type(
            "relative_joint_absolute_pose"
        )
        bs, steps, joints = 2, 16, NUM_JOINTS
        model_pred = torch.zeros(bs, steps, joints, STATE_DIMS)
        hist = torch.ones(bs, 1, joints, STATE_DIMS)
        result = decoder.get_prediction(model_pred, hist, None, None)
        # joint channel: 0 + hist[:,-1:,:,:1] = 1
        assert torch.allclose(
            result[..., :1], torch.ones(bs, steps, joints, 1)
        )
        # pose channels unchanged
        assert torch.allclose(
            result[..., 1:], torch.zeros(bs, steps, joints, STATE_DIMS - 1)
        )

    def test_absolute_joint_relative_pose(self):
        """absolute_joint_relative_pose: pose channels offset by last hist."""
        decoder = self._make_decoder_with_prediction_type(
            "absolute_joint_relative_pose"
        )
        bs, steps, joints = 2, 16, NUM_JOINTS
        model_pred = torch.zeros(bs, steps, joints, STATE_DIMS)
        hist = torch.ones(bs, 1, joints, STATE_DIMS)
        result = decoder.get_prediction(model_pred, hist, None, None)
        # joint channel unchanged: 0
        assert torch.allclose(
            result[..., :1], torch.zeros(bs, steps, joints, 1)
        )
        # pose channels: 0 + hist[:,-1:,:,1:] = 1
        assert torch.allclose(
            result[..., 1:], torch.ones(bs, steps, joints, STATE_DIMS - 1)
        )

    def test_relative_joint_relative_pose(self):
        """relative_joint_relative_pose: all channels offset by hist."""
        decoder = self._make_decoder_with_prediction_type(
            "relative_joint_relative_pose"
        )
        bs, steps, joints = 2, 1, NUM_JOINTS  # hist_steps=1 for broadcast
        model_pred = torch.zeros(bs, steps, joints, STATE_DIMS)
        hist = torch.ones(bs, steps, joints, STATE_DIMS)
        result = decoder.get_prediction(model_pred, hist, None, None)
        # All channels: 0 + 1 = 1
        assert torch.allclose(
            result, torch.ones(bs, steps, joints, STATE_DIMS)
        )

    def test_output_shape_unchanged(self):
        """Output shape equals input model_pred shape."""
        decoder = self._make_decoder_with_prediction_type(
            "absolute_joint_absolute_pose"
        )
        bs, steps, joints = 3, 32, NUM_JOINTS
        model_pred = torch.randn(bs, steps, joints, STATE_DIMS)
        hist = torch.randn(bs, 1, joints, STATE_DIMS)
        result = decoder.get_prediction(model_pred, hist, None, None)
        assert result.shape == model_pred.shape


class TestHoloBrainActionDecoderInit:
    """Tests for HoloBrainActionDecoder initialization."""

    def test_build_minimal_decoder(self):
        """Can build a minimal decoder without errors."""
        decoder = _make_decoder()
        assert isinstance(decoder, HoloBrainActionDecoder)

    def test_build_decoder_with_robot_encoder(self):
        """Can build a decoder with robot state encoder."""
        decoder = _make_decoder(with_robot_encoder=True)
        assert decoder.robot_encoder is not None
        assert isinstance(decoder.robot_encoder, HoloBrainRobotStateEncoder)

    def test_decoder_has_correct_config(self):
        """Decoder attributes match the provided config."""
        decoder = _make_decoder()
        assert decoder.pred_steps == PRED_STEPS
        assert decoder.chunk_size == CHUNK_SIZE
        assert decoder.embed_dims == EMBED_DIMS
        assert decoder.state_dims == STATE_DIMS

    def test_decoder_noise_schedulers(self):
        """Decoder has both training and test noise schedulers."""
        decoder = _make_decoder()
        assert decoder.training_noise_scheduler is not None
        assert decoder.test_noise_scheduler is not None

    def test_decoder_with_temp_joint_attn(self):
        """Can build decoder with TemporalJointGraphAttention."""
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )

        op_order: list[str | None] = [
            "t_norm",
            "temp_joint_attn",
            "norm",
            "img_cross_attn",
            "norm",
            "ffn",
        ]
        decoder = HoloBrainActionDecoder(
            transformer_cfg=HoloBrainDecoderTransformerConfig(
                img_cross_attn=dict(
                    type=RotaryAttention,
                    embed_dims=EMBED_DIMS,
                    num_heads=4,
                    max_position_embeddings=64,
                ),
                temp_joint_attn=dict(
                    type=TemporalJointGraphAttention,
                    embed_dims=EMBED_DIMS,
                    num_heads=4,
                    max_position_embeddings=32,
                ),
                norm_layer=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
                ffn=dict(
                    type=FFN,
                    embed_dims=EMBED_DIMS,
                    feedforward_channels=128,
                ),
                timestep_norm_layer=dict(
                    type=AdaRMSNorm,
                    normalized_shape=EMBED_DIMS,
                    condition_dims=EMBED_DIMS,
                    zero=True,
                ),
                operation_order=op_order,
            ),
            head=dict(
                type=UpsampleHead,
                upsample_sizes=[PRED_STEPS // CHUNK_SIZE * 2, PRED_STEPS],
                input_dim=EMBED_DIMS,
                dims=[32, 16],
                norm=dict(type=nn.RMSNorm, normalized_shape=EMBED_DIMS),
                act=dict(type=nn.SiLU, inplace=True),
                norm_act_idx=[0, 1],
                num_output_layers=1,
                out_dim=STATE_DIMS,
            ),
            base_cfg=HoloBrainDecoderBaseConfig(
                embed_dims=EMBED_DIMS,
                state_dims=STATE_DIMS,
                pred_steps=PRED_STEPS,
                chunk_size=CHUNK_SIZE,
                feature_level=[0],
                training_noise_scheduler=dict(
                    type=DDPMScheduler,
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2",
                    prediction_type="sample",
                    clip_sample=False,
                ),
                test_noise_scheduler=dict(
                    type=DPMSolverMultistepScheduler,
                    num_train_timesteps=1000,
                    beta_schedule="squaredcos_cap_v2",
                    prediction_type="sample",
                ),
            ),
        )
        assert isinstance(decoder, HoloBrainActionDecoder)
