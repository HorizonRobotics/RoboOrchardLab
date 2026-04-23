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

"""Focused tests for GenieSim3 policy pure functions and payload contract."""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from projects.holobrain.policy.geniesim3_policy import (
    GENIESIM_ACTION_DIM,
    GENIESIM_CAMERAS,
    GENIESIM_STATE_DIM,
    TASK_NAME_TO_HEAD_STATE,
    HoloBrainGenieSim3Policy,
    HoloBrainGenieSim3PolicyCfg,
    build_joint_state_from_payload,
    convert_actions_to_geniesim,
)
from robo_orchard_lab.models.holobrain.pipeline import (
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.holobrain.processor import (
    MultiArmManipulationOutput,
)


class _StubModel:
    def eval(self):
        return None


class _StubPipeline:
    def __init__(self, action_steps: int = 8):
        self.cfg = HoloBrainInferencePipelineCfg()
        self.model = _StubModel()
        self.processor = type("Processor", (), {"transforms": []})()
        self.action_steps = action_steps
        self.last_input = None

    def __call__(self, data):
        self.last_input = data
        return MultiArmManipulationOutput(
            action=np.ones(
                (self.action_steps, GENIESIM_STATE_DIM),
                dtype=np.float32,
            )
        )


def _valid_payload(with_depth: bool = False) -> dict:
    images = {
        camera_name: np.zeros((6, 5, 3), dtype=np.uint8)
        for camera_name in GENIESIM_CAMERAS
    }
    payload = {
        "images": images,
        "state": np.arange(21, dtype=np.float32),
        "task_name": "hold_pot",
        "prompt": "custom prompt",
    }
    if with_depth:
        payload["depth"] = {
            camera_name: np.ones((6, 5), dtype=np.float32)
            for camera_name in GENIESIM_CAMERAS
        }
    return payload


class TestBuildJointStateFromPayload:
    """Tests for build_joint_state_from_payload."""

    @pytest.fixture()
    def valid_state(self):
        return np.random.default_rng(42).random(21).astype(np.float32)

    @pytest.mark.parametrize("task_name", list(TASK_NAME_TO_HEAD_STATE.keys()))
    def test_all_canonical_task_names_accepted(self, valid_state, task_name):
        result = build_joint_state_from_payload(valid_state, task_name)
        assert result.shape == (GENIESIM_STATE_DIM,)
        assert result.dtype == np.float32

    def test_unknown_task_name_raises(self, valid_state):
        with pytest.raises(ValueError, match="Unknown GenieSim3 task_name"):
            build_joint_state_from_payload(valid_state, "nonexistent_task")

    def test_short_state_raises(self):
        short_state = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="at least 21 dims"):
            build_joint_state_from_payload(short_state, "hold_pot")

    def test_head_state_populated_from_task(self, valid_state):
        result = build_joint_state_from_payload(valid_state, "hold_pot")
        expected_head = np.array(
            TASK_NAME_TO_HEAD_STATE["hold_pot"], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result[16:19], expected_head)

    def test_arm_joints_mapped_correctly(self, valid_state):
        result = build_joint_state_from_payload(valid_state, "hold_pot")
        np.testing.assert_array_almost_equal(result[:7], valid_state[:7])
        np.testing.assert_array_almost_equal(result[8:15], valid_state[7:14])

    def test_gripper_states_use_training_observation_normalization(self):
        state = np.zeros(21, dtype=np.float32)
        state[14] = 120.0
        state[15] = 30.0

        result = build_joint_state_from_payload(state, "hold_pot")

        assert result[7] == pytest.approx(1.0)
        assert result[15] == pytest.approx(0.25)

    def test_body_joints_mapped(self, valid_state):
        result = build_joint_state_from_payload(valid_state, "hold_pot")
        np.testing.assert_array_almost_equal(result[19:24], valid_state[16:21])


class TestConvertActionsToGeniesim:
    """Tests for convert_actions_to_geniesim."""

    def test_output_shape(self):
        actions = (
            np.random.default_rng(42)
            .random((64, GENIESIM_STATE_DIM))
            .astype(np.float32)
        )
        result = convert_actions_to_geniesim(actions, valid_action_step=32)
        assert result.shape == (32, GENIESIM_ACTION_DIM)
        assert result.dtype == np.float32

    def test_batch_dim_squeezed(self):
        actions = (
            np.random.default_rng(42)
            .random((1, 64, GENIESIM_STATE_DIM))
            .astype(np.float32)
        )
        result = convert_actions_to_geniesim(actions, valid_action_step=32)
        assert result.shape == (32, GENIESIM_ACTION_DIM)

    def test_invalid_action_dim_raises(self):
        actions = np.zeros((64, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected action array"):
            convert_actions_to_geniesim(actions, valid_action_step=32)

    def test_zero_valid_action_step_raises(self):
        actions = np.zeros((64, GENIESIM_STATE_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="valid_action_step must be > 0"):
            convert_actions_to_geniesim(actions, valid_action_step=0)

    def test_sampling_ratio_stride(self):
        actions = (
            np.random.default_rng(42)
            .random((64, GENIESIM_STATE_DIM))
            .astype(np.float32)
        )
        result = convert_actions_to_geniesim(
            actions, valid_action_step=16, sampling_ratio=2.0
        )
        assert result.shape == (16, GENIESIM_ACTION_DIM)

    def test_identity_sampling_ratio(self):
        actions = (
            np.random.default_rng(42)
            .random((64, GENIESIM_STATE_DIM))
            .astype(np.float32)
        )
        r1 = convert_actions_to_geniesim(
            actions, valid_action_step=32, sampling_ratio=1.0
        )
        r2 = convert_actions_to_geniesim(actions, valid_action_step=32)
        np.testing.assert_array_equal(r1, r2)


class TestHoloBrainGenieSim3PolicyPayloadContract:
    """Tests for the documented GenieSim3 deploy payload contract."""

    def test_data_preprocess_accepts_no_depth_payload(self):
        policy = HoloBrainGenieSim3Policy(
            cfg=HoloBrainGenieSim3PolicyCfg(
                use_depth=False,
                task_name_to_instruction={},
            ),
            pipeline=_StubPipeline(),
        )
        data = policy.data_preprocess(_valid_payload())

        assert set(data.image.keys()) == set(GENIESIM_CAMERAS)
        assert set(data.depth.keys()) == set(GENIESIM_CAMERAS)
        assert data.instruction == "custom prompt"
        assert data.history_joint_state[0].shape == (GENIESIM_STATE_DIM,)
        for depth_list in data.depth.values():
            assert depth_list[0].shape == (6, 5)
            assert np.all(depth_list[0] == 0)

    def test_data_preprocess_prefers_configured_task_instruction(self):
        policy = HoloBrainGenieSim3Policy(
            cfg=HoloBrainGenieSim3PolicyCfg(use_depth=False),
            pipeline=_StubPipeline(),
        )

        data = policy.data_preprocess(_valid_payload())

        assert (
            data.instruction == policy.cfg.task_name_to_instruction["hold_pot"]
        )
        assert data.instruction != "custom prompt"

    def test_data_preprocess_requires_depth_when_enabled(self):
        policy = HoloBrainGenieSim3Policy(
            cfg=HoloBrainGenieSim3PolicyCfg(use_depth=True),
            pipeline=_StubPipeline(),
        )

        with pytest.raises(ValueError, match="missing depth"):
            policy.data_preprocess(_valid_payload())

    def test_data_preprocess_rejects_missing_image_camera(self):
        policy = HoloBrainGenieSim3Policy(
            cfg=HoloBrainGenieSim3PolicyCfg(use_depth=False),
            pipeline=_StubPipeline(),
        )
        payload = _valid_payload()
        del payload["images"]["hand_left"]

        with pytest.raises(ValueError, match="missing image"):
            policy.data_preprocess(payload)

    def test_get_actions_returns_geniesim_action_shape(self):
        pipeline = _StubPipeline(action_steps=8)
        policy = HoloBrainGenieSim3Policy(
            cfg=HoloBrainGenieSim3PolicyCfg(
                use_depth=False,
                valid_action_step=4,
            ),
            pipeline=pipeline,
        )

        actions = policy.get_actions(_valid_payload())

        assert pipeline.last_input is not None
        assert actions.shape == (4, GENIESIM_ACTION_DIM)
        assert actions.dtype == np.float32


class TestGenieSim3ConfigBuilder:
    """Tests for the shipped GenieSim3 config builder surface."""

    def test_build_datasets_forwards_pred_interval(self, monkeypatch):
        repo_root = Path(__file__).resolve().parents[4]
        config_dir = repo_root / "projects" / "holobrain" / "configs"
        monkeypatch.syspath_prepend(str(config_dir))
        cfg_module = importlib.import_module("config_agibot_geniesim_dataset")

        captured_kwargs = {}

        class _FakeAgibotGenieSim3RODataset:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        fake_dataset_module = types.ModuleType(
            "robo_orchard_lab.dataset.agibot_geniesim."
            "agibot_geniesim3_ro_dataset"
        )
        fake_dataset_module.AgibotGenieSim3RODataset = (
            _FakeAgibotGenieSim3RODataset
        )
        monkeypatch.setitem(
            sys.modules,
            fake_dataset_module.__name__,
            fake_dataset_module,
        )
        monkeypatch.setattr(cfg_module, "pred_interval", 7)
        monkeypatch.setattr(
            cfg_module,
            "expand_ro_data_paths",
            lambda patterns: ["packed-shard"],
        )
        monkeypatch.setattr(
            cfg_module,
            "build_transforms",
            lambda *args, **kwargs: [],
        )

        datasets = cfg_module.build_datasets(
            {"hist_steps": 2, "pred_steps": 5},
            [cfg_module.dataset_name],
            mode="training",
        )

        assert len(datasets) == 1
        assert captured_kwargs["paths"] == ["packed-shard"]
        assert captured_kwargs["pred_interval"] == 7


class _FakePolicy:
    def __init__(self, exc: Exception):
        self.cfg = type("Cfg", (), {"valid_action_step": 3})()
        self.exc = exc

    def get_actions(self, payload):
        raise self.exc


class _FakeWebsocket:
    def __init__(self, messages: list[bytes]):
        self._messages = list(messages)
        self.sent: list[bytes] = []

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, message: bytes) -> None:
        self.sent.append(message)


class TestGenieSim3WebsocketServer:
    """Tests for websocket response error signaling."""

    def test_exception_response_has_error_and_zero_actions(self):
        from projects.holobrain.scripts.geniesim3_inference_server import (
            HoloBrainGenieSim3WebsocketServer,
            packb,
            unpackb,
        )

        websocket = _FakeWebsocket([packb({"bad": "payload"})])
        server = HoloBrainGenieSim3WebsocketServer(
            _FakePolicy(RuntimeError("boom")),
            host="127.0.0.1",
            port=8999,
        )

        asyncio.run(server.handler(websocket))

        assert len(websocket.sent) == 2
        metadata = unpackb(websocket.sent[0])
        response = unpackb(websocket.sent[1])
        assert metadata["valid_action_step"] == 3
        assert response["error"] == "RuntimeError: boom"
        assert response["request_count"] == 1
        assert response["actions"].shape == (3, GENIESIM_ACTION_DIM)
        assert np.all(response["actions"] == 0)
