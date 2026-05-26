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

"""Focused tests for internal GenieSim3 deploy payload contracts."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMMON_ROOT = _REPO_ROOT / "projects" / "holobrain_internal" / "common"
sys.path.insert(0, str(_COMMON_ROOT))

from holobrain_geniesim3_policy import (  # noqa: E402
    GENIESIM_ACTION_DIM,
    GENIESIM_CAMERAS,
    GENIESIM_STATE_DIM,
    HoloBrainGenieSim3Policy,
    HoloBrainGenieSim3PolicyCfg,
    build_joint_state_from_payload,
    convert_actions_to_geniesim,
    deploy_policy as geniesim3_deploy,  # noqa: E402
)
from holobrain_geniesim3_policy.deploy_policy import (  # noqa: E402
    TASK_NAME_TO_HEAD_STATE,
)

from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    MultiArmManipulationOutput,
)


class _StubModel:
    def eval(self):
        return None


class _StubPipeline:
    def __init__(self, action_steps: int = 8):
        self.model = _StubModel()
        self.processor = type("Processor", (), {"transforms": []})()
        self.action_steps = action_steps
        self.last_input = None

    def __call__(self, data):
        self.last_input = data
        action = np.ones(
            (self.action_steps, GENIESIM_STATE_DIM),
            dtype=np.float32,
        )
        return MultiArmManipulationOutput(
            action=action,
            pose=action[..., None],
        )


class _LoadedProcessor:
    transforms = []


class _LoadedModel:
    def eval(self):
        return None

    def requires_grad_(self, requires_grad: bool):
        return None

    def to(self, device):
        return None


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


def test_build_joint_state_maps_geniesim_payload_order():
    state = np.arange(21, dtype=np.float32)
    state[14] = 120.0
    state[15] = 30.0

    result = build_joint_state_from_payload(state, "hold_pot")

    assert result.shape == (GENIESIM_STATE_DIM,)
    np.testing.assert_array_equal(result[:7], state[:7])
    np.testing.assert_array_equal(result[8:15], state[7:14])
    assert result[7] == pytest.approx(1.0)
    assert result[15] == pytest.approx(0.25)
    np.testing.assert_array_equal(
        result[16:19],
        np.asarray(TASK_NAME_TO_HEAD_STATE["hold_pot"], dtype=np.float32),
    )
    np.testing.assert_array_equal(result[19:24], state[16:21])


def test_build_joint_state_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unknown GenieSim3 task_name"):
        build_joint_state_from_payload(np.zeros(21), "missing_task")


def test_convert_actions_to_geniesim_shape_and_layout():
    actions = np.zeros((64, GENIESIM_STATE_DIM), dtype=np.float32)
    actions[:, 0:7] = 1
    actions[:, 8:15] = 2
    actions[:, 7] = 3
    actions[:, 15] = 4
    actions[:, 23] = 5

    result = convert_actions_to_geniesim(actions, valid_action_step=32)

    assert result.shape == (32, GENIESIM_ACTION_DIM)
    np.testing.assert_array_equal(result[:, :7], 1)
    np.testing.assert_array_equal(result[:, 7:14], 2)
    np.testing.assert_array_equal(result[:, 14], 3)
    np.testing.assert_array_equal(result[:, 15], 4)
    np.testing.assert_array_equal(result[:, 20], 5)


def test_data_preprocess_accepts_no_depth_payload():
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


def test_data_preprocess_requires_depth_when_enabled():
    policy = HoloBrainGenieSim3Policy(
        cfg=HoloBrainGenieSim3PolicyCfg(use_depth=True),
        pipeline=_StubPipeline(),
    )

    with pytest.raises(ValueError, match="missing depth"):
        policy.data_preprocess(_valid_payload())


def test_get_actions_returns_geniesim_action_shape():
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


def test_inference_server_parse_args_accepts_explicit_use_depth_bool():
    from geniesim3_inference_server import parse_args, str2bool

    assert parse_args([]).use_depth is False
    assert parse_args(["--use_depth", "true"]).use_depth is True
    assert parse_args(["--use_depth", "false"]).use_depth is False
    assert parse_args(["--use_depth", "True"]).use_depth is True
    assert parse_args(["--use_depth", "False"]).use_depth is False
    assert parse_args(["--use_depth", "1"]).use_depth is True
    assert parse_args(["--use_depth", "0"]).use_depth is False
    assert str2bool(True) is True


def test_inference_server_parse_args_rejects_no_use_depth_flag():
    from geniesim3_inference_server import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--no-use_depth"])


def test_http_model_dir_downloads_checkpoint_and_links_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "urdf").mkdir()
    cache_dir = tmp_path / "cache"
    downloaded: list[tuple[str, Path]] = []
    loaded_model_dirs: list[str] = []

    def fake_download_file(url: str, file_name: str) -> None:
        target = Path(file_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        downloaded.append((url, target))

    def fake_load_processor(path: str, processor_name: str):
        assert path == str(cache_dir)
        assert processor_name == "agibot_geniesim3_challenge_processor.json"
        return _LoadedProcessor()

    def fake_load_model(directory: str, **kwargs):
        loaded_model_dirs.append(directory)
        assert kwargs["model_prefix"] == "model"
        return _LoadedModel()

    monkeypatch.setattr(
        geniesim3_deploy, "DEFAULT_MODEL_CACHE_DIR", str(cache_dir)
    )
    monkeypatch.setattr(geniesim3_deploy, "download_file", fake_download_file)
    monkeypatch.setattr(
        geniesim3_deploy.HoloBrainProcessor,
        "load",
        staticmethod(fake_load_processor),
    )
    monkeypatch.setattr(
        geniesim3_deploy.ModelMixin,
        "load_model",
        staticmethod(fake_load_model),
    )

    cfg = HoloBrainGenieSim3PolicyCfg(
        model_dir=("http://host/user/run/output/checkpoints/checkpoint_9/"),
        model_processor="agibot_geniesim3_challenge_processor",
    )
    HoloBrainGenieSim3Policy(cfg=cfg)

    assert [url for url, _ in downloaded] == [
        "http://host/user/run/output/checkpoints/checkpoint_9/model.safetensors",
        "http://host/user/run/output/checkpoints/checkpoint_9/model.config.json",
        "http://host/user/run/output/agibot_geniesim3_challenge_processor.json",
    ]
    assert cfg.model_dir == str(cache_dir)
    assert loaded_model_dirs == [str(cache_dir)]
    assert (cache_dir / "ckpt").resolve() == (tmp_path / "ckpt")
    assert (cache_dir / "urdf").resolve() == (tmp_path / "urdf")


def test_load_processor_finds_local_checkpoint_grandparent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "output"
    checkpoint_dir = output_dir / "checkpoints" / "checkpoint_9"
    checkpoint_dir.mkdir(parents=True)
    (output_dir / "custom_processor.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, str]] = []

    def fake_load_processor(path: str, processor_name: str):
        calls.append((path, processor_name))
        return _LoadedProcessor()

    monkeypatch.setattr(
        geniesim3_deploy.HoloBrainProcessor,
        "load",
        staticmethod(fake_load_processor),
    )

    policy = HoloBrainGenieSim3Policy(
        cfg=HoloBrainGenieSim3PolicyCfg(model_processor="custom_processor"),
        pipeline=_StubPipeline(),
    )

    processor = policy._load_processor(str(checkpoint_dir))

    assert isinstance(processor, _LoadedProcessor)
    assert calls == [(str(output_dir), "custom_processor.json")]


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


def test_websocket_exception_response_has_error_and_zero_actions():
    from geniesim3_inference_server import (
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
