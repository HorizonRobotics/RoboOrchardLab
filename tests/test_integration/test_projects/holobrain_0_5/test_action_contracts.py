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


"""Regression coverage for migrated action, batching and frame contracts."""

import pytest
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion
from robo_orchard_core.utils.config import ClassType_co
from safetensors.torch import save_model

from projects.holobrain_0_5 import utils
from projects.holobrain_0_5.configs.dataset_factory import collate_action_batch
from projects.holobrain_0_5.policy.libero_policy import apply_transform
from projects.holobrain_0_5.utils import load_checkpoint
from robo_orchard_lab.models.holobrain_0_5.layers import (
    ActionFFN,
    ScalarEmbedder,
)
from robo_orchard_lab.models.holobrain_0_5.pipeline import (
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.layers.transformer_layers import FFN
from robo_orchard_lab.models.mixin import ModelMixin, TorchModuleCfg

__all__ = []


class _DropoutModel(ModelMixin):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.dropout = torch.nn.Dropout(p=1.0)

    def forward(self, data):
        return self.dropout(data["input"]) * self.scale


class _DropoutModelCfg(TorchModuleCfg[_DropoutModel]):
    class_type: ClassType_co[_DropoutModel] = _DropoutModel


class TestActionContracts:
    def test_exported_pipeline_uses_evaluation_mode(self, tmp_path):
        model = _DropoutModelCfg()()
        pipeline = HoloBrainInferencePipeline(
            HoloBrainInferencePipelineCfg(), model
        )
        pipeline.save_pipeline(str(tmp_path))
        restored = HoloBrainInferencePipeline.load_pipeline(
            str(tmp_path), load_impl="native"
        )
        data = {"input": torch.ones(2, 3)}
        torch.testing.assert_close(restored(data), data["input"])

    def test_collate_scalars_keeps_sample_order(self):
        samples = [
            dict(
                arm_length=torch.tensor(x),
                imgs=torch.full((2, 3, 4, 3), x),
                text=str(x),
            )
            for x in [1.0, 2.0]
        ]
        output = collate_action_batch(samples)
        torch.testing.assert_close(
            output["arm_length"], torch.tensor([1.0, 2.0])
        )
        assert output["imgs"].shape == (2, 2, 3, 4, 3)
        assert output["text"] == ["1.0", "2.0"]
        assert samples[0]["arm_length"].ndim == 0

    def test_action_ffn_zero_residual_and_checkpoint_layout(self):
        action = ActionFFN(embed_dims=8, feedforward_channels=16)
        original = FFN(embed_dims=8, feedforward_channels=16)
        x = torch.randn(2, 3, 8)
        torch.testing.assert_close(action(x), x)
        action.load_state_dict(original.state_dict(), strict=True)
        torch.testing.assert_close(action(x), original(x))

    def test_checkpoint_rejects_wrong_architecture(self, tmp_path):
        original = torch.nn.Linear(3, 4)
        path = tmp_path / "model.safetensors"
        save_model(original, str(path))
        restored = torch.nn.Linear(3, 4)
        load_checkpoint(restored, str(path))
        torch.testing.assert_close(restored.weight, original.weight)
        with pytest.raises(RuntimeError):
            load_checkpoint(torch.nn.Linear(3, 5), str(path))

    def test_checkpoint_accepts_hf_uri(self, tmp_path, monkeypatch):
        original = torch.nn.Linear(3, 4)
        path = tmp_path / "model.safetensors"
        save_model(original, str(path))
        uri = "hf://model/Org/Repo/pretrain/model.safetensors"
        monkeypatch.setattr(
            utils, "resolve_hf_compatible_path", lambda value: str(path)
        )
        restored = torch.nn.Linear(3, 4)

        load_checkpoint(restored, uri)

        torch.testing.assert_close(restored.weight, original.weight)

    def test_checkpoint_error_reports_resolved_location(
        self, tmp_path, monkeypatch
    ):
        missing = tmp_path / "absent.safetensors"
        uri = "hf://model/Org/Repo/pretrain/model.safetensors"
        monkeypatch.setattr(
            utils, "resolve_hf_compatible_path", lambda value: str(missing)
        )

        with pytest.raises(FileNotFoundError) as excinfo:
            load_checkpoint(torch.nn.Linear(3, 4), uri)

        message = str(excinfo.value)
        assert uri in message
        assert str(missing) in message

    def test_legacy_frequency_buffer_is_derived(self):
        embedding = ScalarEmbedder(16)
        x = torch.tensor([0.0, 1.0])
        expected = embedding(x)
        weights = dict(embedding.state_dict(), freqs=embedding.freqs.clone())
        embedding.load_state_dict(weights, strict=True)
        torch.testing.assert_close(embedding(x), expected)

    def test_libero_frame_round_trip(self):
        # Non-commuting rotations catch left/right multiplication mistakes.
        base_to_ego = torch.eye(4, dtype=torch.float64)
        base_to_ego[:3, :3] = axis_angle_to_matrix(
            torch.tensor([0.2, 0.3, -0.1], dtype=torch.float64)
        )
        base_to_ego[:3, 3] = torch.tensor([0.4, 0.2, 0.8])
        alignment = torch.eye(4, dtype=torch.float64)
        alignment[:3, :3] = axis_angle_to_matrix(
            torch.tensor([0.0, 0.0, 0.7], dtype=torch.float64)
        )
        state = torch.tensor(
            [[[0.25, 0.3, -0.2, 0.1, 1.0, 0, 0, 0]]], dtype=torch.float64
        )
        world_pose = torch.eye(4, dtype=torch.float64)
        world_pose[:3, 3] = state[0, 0, 1:4]
        ego_pose = base_to_ego @ world_pose @ alignment
        transformed = state.clone()
        transformed[..., 1:4] = ego_pose[:3, 3]
        transformed[..., 4:8] = matrix_to_quaternion(ego_pose[:3, :3])
        restored = apply_transform(transformed, base_to_ego, alignment)
        torch.testing.assert_close(restored, state)
