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

import importlib.util
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
from PIL import Image

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "projects"
    / "holobrain_internal"
    / "common"
    / "data_visualize"
    / "app.py"
)


def _load_app_module():
    spec = importlib.util.spec_from_file_location(
        "holobrain_internal_data_visualize_app", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app_module = _load_app_module()


def _frame_payload(episode_idx, index, *, dataset_name=None, step_index=None):
    label = dataset_name if dataset_name is not None else str(episode_idx)
    return {
        "uuid": f"uuid-{label}",
        "text": f"instruction-{label}-{index}",
        "subtask": f"subtask-{label}-{index}",
        "step_index": index * 10 if step_index is None else step_index,
        "imgs": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "depths": np.zeros((1, 2, 2), dtype=np.float32),
    }


class _FakeDataset:
    dataset_name = "fake_dataset"
    num_episode = 2

    def __init__(self):
        self._ranges = [(0, 5), (5, 8)]

    def get_episode_range(self, ep_idx):
        return self._ranges[ep_idx]

    def __getitem__(self, index):
        episode_idx = 0 if index < self._ranges[1][0] else 1
        return _frame_payload(episode_idx, index)


class _FakeSingleEpisodeDataset:
    def __init__(self, dataset_name, frame_base):
        self.dataset_name = dataset_name
        self.num_episode = 1
        self.frame_base = frame_base

    def __len__(self):
        return 2

    def get_episode_range(self, ep_idx):
        if ep_idx != 0:
            raise IndexError
        return 0, 2

    def __getitem__(self, index):
        value = self.frame_base + index
        return _frame_payload(
            None,
            index,
            dataset_name=self.dataset_name,
            step_index=value,
        )


class _FakeRangeOnlyDataset:
    def __init__(self, dataset_name, frame_base):
        self.dataset_name = dataset_name
        self.frame_base = frame_base

    def __len__(self):
        return 2

    def get_episode_range(self, ep_idx):
        if ep_idx != 0:
            raise IndexError
        return 0, 2

    def __getitem__(self, index):
        value = self.frame_base + index
        return _frame_payload(
            None,
            index,
            dataset_name=self.dataset_name,
            step_index=value,
        )


class _FakeEpisodeIndexDataset:
    dataset_name = "episode_index_dataset"

    def __init__(self):
        self._episode_indices = [0, 0, 1, 1, 1]

    def __len__(self):
        return len(self._episode_indices)

    def __getitem__(self, index):
        if index == "episode_index":
            return self._episode_indices
        episode_idx = self._episode_indices[index]
        return _frame_payload(episode_idx, index)


class _FakeIndexedDataset:
    def __init__(self, dataset_name, episode_indices, frame_base):
        self.dataset_name = dataset_name
        self.index_dataset = {"episode_index": episode_indices}
        self.frame_base = frame_base

    def __len__(self):
        return len(self.index_dataset["episode_index"])

    def __getitem__(self, index):
        episode_idx = self.index_dataset["episode_index"][index]
        return _frame_payload(
            episode_idx,
            index,
            dataset_name=self.dataset_name,
            step_index=self.frame_base + index,
        )


class _FakeVisualizer:
    def _render_frame(self, data, ee_indices):
        value = 50 + len(tuple(ee_indices))
        return np.full((4, 4, 3), value, dtype=np.uint8)


class _CountingVisualizer:
    def __init__(self):
        self.render_count = 0
        self.lock = threading.Lock()

    def _render_frame(self, data, ee_indices):
        time.sleep(0.01)
        with self.lock:
            self.render_count += 1
        return np.full((4, 4, 3), 80, dtype=np.uint8)


def _make_handle(**overrides):
    values = {
        "dataset_id": 0,
        "name": "fake_dataset",
        "dataset": _FakeDataset(),
        "visualizer": _FakeVisualizer(),
        "datasets": {"training": _FakeDataset()},
        "visualizers": {"training": _FakeVisualizer()},
        "lock": threading.Lock(),
    }
    values.update(overrides)
    return app_module.DatasetHandle(**values)


def _make_client(*, handle=None, **app_kwargs):
    settings = {
        "dataset_handles": [handle or _make_handle()],
        "fps": 12,
        "interval": 1,
        "max_cache_frames": 4,
    }
    settings.update(app_kwargs)
    return app_module.create_app(**settings).test_client()


def _install_config(monkeypatch, config):
    monkeypatch.setattr(
        app_module,
        "load_config",
        lambda _path: config,
    )


def _fake_concat(*datasets):
    return type("_FakeConcat", (), {"datasets": list(datasets)})()


def _assert_error_response(response, status_code):
    assert response.status_code == status_code
    assert "error" in response.get_json()


def _make_counting_client(visualizer):
    handle = _make_handle(
        dataset_id=0,
        name="fake_dataset",
        dataset=_FakeDataset(),
        visualizer=visualizer,
        datasets={"training": _FakeDataset()},
        visualizers={"training": visualizer},
    )
    return _make_client(
        handle=handle,
        max_cache_frames=8,
        prefetch_frames=2,
        prefetch_workers=2,
    )


def test_datasets_endpoint_returns_selectable_datasets():
    client = _make_client()

    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert response.get_json() == {
        "datasets": [
            {
                "id": 0,
                "name": "fake_dataset",
                "loaded_modes": ["training"],
                "num_episodes": {"training": 2},
            }
        ]
    }


def test_index_renders_dataset_options_without_client_fetch():
    client = _make_client()

    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "<h1>HoloBrain Data Visualizer</h1>" in html
    assert '<option value="0">fake_dataset (loaded: training)</option>' in html
    assert 'id="load"' in html
    assert 'id="copyMeta"' in html
    assert 'id="metadataJson"' in html
    assert 'id="loadMeta"' in html
    assert "Copy JSON" in html
    assert "Load From JSON" in html
    assert "__DATASET_OPTIONS__" not in html
    assert "__INITIAL_DATASETS__" not in html


def test_index_script_has_valid_javascript(tmp_path):
    if shutil.which("node") is None:
        return
    client = _make_client()

    response = client.get("/")

    scripts = re.findall(
        r"<script>(.*?)</script>", response.get_data(as_text=True), re.S
    )
    assert scripts
    script_path = tmp_path / "data_visualize_app.js"
    script_path.write_text("\n".join(scripts), encoding="utf-8")
    subprocess.run(
        ["node", "--check", str(script_path)],
        check=True,
        text=True,
        capture_output=True,
    )


def test_build_dataset_handles_lists_all_configured_dataset_candidates(
    monkeypatch,
):
    class _FakeConfig:
        config = {
            "training_datasets": ["train_a"],
            "validation_datasets": ["val_c"],
            "deploy_datasets": ["deploy_b"],
        }

        @staticmethod
        def build_training_dataset(_config):
            return None

        @staticmethod
        def build_validation_dataset(_config):
            return None

    _install_config(monkeypatch, _FakeConfig)

    handles = app_module.build_dataset_handles(
        "fake_config.py",
        dataset_names=None,
    )

    assert [handle.name for handle in handles] == [
        "deploy_b",
        "train_a",
        "val_c",
    ]


def test_build_dataset_handles_uses_explicit_dataset_names_as_candidates(
    monkeypatch,
):
    class _FakeConfig:
        config = {"training_datasets": ["robotwin2_0"]}

        @staticmethod
        def build_training_dataset(_config):
            return None

        @staticmethod
        def build_validation_dataset(_config):
            return None

    _install_config(monkeypatch, _FakeConfig)

    handles = app_module.build_dataset_handles(
        "fake_config.py",
        dataset_names=["robotwin"],
    )

    assert [handle.name for handle in handles] == ["robotwin"]


def test_parse_args_rejects_removed_vis_validation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["data_visualize_app.py", "--config", "fake.py", "--vis_validation"],
    )

    with pytest.raises(SystemExit) as exc_info:
        app_module.parse_args()

    assert exc_info.value.code != 0


def test_datasets_endpoint_does_not_build_lazy_dataset():
    build_calls = []

    def build_dataset(config):
        build_calls.append(config["training_datasets"])
        return _fake_concat(_FakeDataset())

    handle = _make_handle(
        name="lazy_dataset",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"training_datasets": ["lazy_dataset"]},
        build_training_dataset=build_dataset,
    )
    client = _make_client(handle=handle)

    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert response.get_json() == {
        "datasets": [
            {
                "id": 0,
                "name": "lazy_dataset",
                "loaded_modes": [],
                "num_episodes": {},
            }
        ]
    }
    assert build_calls == []

    episode_response = client.get("/api/episodes/0/0")

    assert episode_response.status_code == 200
    assert episode_response.get_json()["dataset_name"] == "lazy_dataset"
    assert episode_response.get_json()["num_episodes"] == 2
    assert build_calls == [["lazy_dataset"]]


def test_validation_mode_can_be_selected_and_built_lazily():
    build_calls = []

    def build_validation_dataset(config):
        build_calls.append(config["validation_datasets"])
        return _fake_concat(_FakeDataset())

    handle = _make_handle(
        name="val_dataset",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"validation_datasets": ["val_dataset"]},
        build_validation_dataset=build_validation_dataset,
    )
    client = _make_client(handle=handle)

    response = client.get("/api/episodes/0/0?mode=validation")

    assert response.status_code == 200
    assert response.get_json()["mode"] == "validation"
    assert build_calls == [["val_dataset"]]


def test_unsupported_mode_returns_client_error():
    handle = _make_handle(
        name="train_only",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"training_datasets": ["train_only"]},
        build_training_dataset=lambda _config: None,
    )
    client = _make_client(handle=handle)

    response = client.get("/api/episodes/0/0?mode=validation")

    _assert_error_response(response, 400)


def test_lazy_dataset_selector_can_expand_to_multiple_datasets():
    handle = _make_handle(
        name="expanded",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"training_datasets": ["expanded"]},
        build_training_dataset=lambda _config: _fake_concat(
            _FakeSingleEpisodeDataset("first", 100),
            _FakeSingleEpisodeDataset("second", 200),
        ),
    )
    client = _make_client(handle=handle)

    episode_response = client.get("/api/episodes/0/1")
    frame_response = client.get("/api/frames/0/1/0")

    assert episode_response.status_code == 200
    assert episode_response.get_json()["uuid"] == "uuid-second"
    assert frame_response.status_code == 200
    assert frame_response.headers["X-Step-Id"] == "1"


def test_lazy_dataset_selector_accepts_datasets_without_num_episode():
    handle = _make_handle(
        name="expanded",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"training_datasets": ["expanded"]},
        build_training_dataset=lambda _config: _fake_concat(
            _FakeRangeOnlyDataset("first", 100),
            _FakeRangeOnlyDataset("second", 200),
        ),
    )
    client = _make_client(handle=handle)

    episode_response = client.get("/api/episodes/0/1")
    frame_response = client.get("/api/frames/0/1/0")

    assert episode_response.status_code == 200
    assert episode_response.get_json()["uuid"] == "uuid-second"
    assert frame_response.status_code == 200
    assert frame_response.headers["X-Data-Index"] == "3"


def test_lazy_dataset_selector_wraps_episode_index_datasets():
    dataset = _FakeEpisodeIndexDataset()
    handle = _make_handle(
        name="expanded",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"training_datasets": ["expanded"]},
        build_training_dataset=lambda _config: _fake_concat(dataset),
    )
    client = _make_client(handle=handle)

    episode_response = client.get("/api/episodes/0/1")
    frame_response = client.get("/api/frames/0/1/1")

    assert episode_response.status_code == 200
    assert episode_response.get_json()["num_episodes"] == 2
    assert episode_response.get_json()["num_frames"] == 2
    assert frame_response.status_code == 200
    assert frame_response.headers["X-Data-Index"] == "4"
    assert unquote(frame_response.headers["X-Instruction"]) == (
        "instruction-1-4"
    )


def test_lazy_dataset_selector_flattens_nested_concat_datasets():
    dataset = _fake_concat(
        _fake_concat(
            _FakeIndexedDataset("first", [0, 0, 1], 100),
            _FakeIndexedDataset("second", [0, 1, 1], 200),
        )
    )
    handle = _make_handle(
        name="expanded",
        dataset=None,
        visualizer=None,
        datasets={},
        visualizers={},
        config={"training_datasets": ["expanded"]},
        build_training_dataset=lambda _config: dataset,
    )
    client = _make_client(handle=handle)

    episode_response = client.get("/api/episodes/0/3")
    frame_response = client.get("/api/frames/0/3/0")

    assert episode_response.status_code == 200
    assert episode_response.get_json()["dataset_name"] == "expanded"
    assert episode_response.get_json()["num_episodes"] == 4
    assert frame_response.status_code == 200
    assert frame_response.headers["X-Data-Index"] == "5"
    assert frame_response.headers["X-Step-Id"] == "1"


def test_episode_endpoint_returns_metadata():
    client = _make_client()

    response = client.get("/api/episodes/0/1")

    assert response.status_code == 200
    assert response.get_json() == {
        "dataset_id": 0,
        "dataset_name": "fake_dataset",
        "mode": "training",
        "episode_id": 1,
        "uuid": "uuid-1",
        "instruction": "instruction-1-6",
        "subtask": "subtask-1-6",
        "num_frames": 2,
        "num_episodes": 2,
        "fps": 12,
        "interval": 1,
        "first_step_id": 1,
    }


def test_frame_endpoint_returns_encoded_image_and_step_id():
    client = _make_client()

    response = client.get("/api/frames/0/1/1")

    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.headers["X-Step-Id"] == "2"
    assert response.headers["X-Data-Index"] == "7"
    assert unquote(response.headers["X-Instruction"]) == "instruction-1-7"
    assert unquote(response.headers["X-Subtask"]) == "subtask-1-7"
    assert response.data.startswith(b"\xff\xd8")


def test_frame_endpoint_uses_requested_highlight_joint_indices():
    client = _make_client()

    response = client.get("/api/frames/0/1/1?highlight_joint_indices=1,2,3")

    assert response.status_code == 200
    image = Image.open(BytesIO(response.data))
    assert image.getpixel((0, 0)) == (53, 53, 53)


def test_frame_endpoint_rejects_invalid_highlight_joint_indices():
    client = _make_client()

    response = client.get("/api/frames/0/1/1?highlight_joint_indices=1,a")

    _assert_error_response(response, 400)


def test_frame_locate_endpoint_returns_offset_for_data_index():
    client = _make_client()

    response = client.get("/api/frames/0/1/locate?data_index=7&vis_interval=1")

    assert response.status_code == 200
    assert response.get_json() == {
        "dataset_id": 0,
        "mode": "training",
        "episode_id": 1,
        "data_index": 7,
        "frame_offset": 1,
    }


def test_frame_locate_endpoint_rejects_data_index_outside_episode():
    client = _make_client()

    response = client.get("/api/frames/0/1/locate?data_index=4&vis_interval=1")

    _assert_error_response(response, 404)


def test_frame_metadata_identifies_original_dataset_item():
    dataset = _FakeDataset()
    client = _make_client(
        handle=_make_handle(
            dataset=dataset,
            datasets={"training": dataset},
        )
    )

    response = client.get("/api/frames/0/1/1")

    data_index = int(response.headers["X-Data-Index"])
    raw_frame = dataset[data_index]
    copied_payload = {
        "dataset_name": "fake_dataset",
        "uuid": raw_frame["uuid"],
        "episode_id": 1,
        "step_id": response.headers["X-Step-Id"],
        "data_index": data_index,
    }
    assert copied_payload == {
        "dataset_name": "fake_dataset",
        "uuid": "uuid-1",
        "episode_id": 1,
        "step_id": "2",
        "data_index": 7,
    }
    assert raw_frame["text"] == unquote(response.headers["X-Instruction"])
    assert raw_frame["subtask"] == unquote(response.headers["X-Subtask"])


def test_vis_interval_skips_frames_from_first_played_frame():
    client = _make_client()

    episode_response = client.get("/api/episodes/0/0?vis_interval=2")
    first = client.get("/api/frames/0/0/0?vis_interval=2")
    second = client.get("/api/frames/0/0/1?vis_interval=2")

    assert episode_response.status_code == 200
    assert episode_response.get_json()["num_frames"] == 2
    assert episode_response.get_json()["interval"] == 2
    assert first.headers["X-Step-Id"] == "1"
    assert second.headers["X-Step-Id"] == "3"
    assert unquote(first.headers["X-Instruction"]) == "instruction-0-1"
    assert unquote(second.headers["X-Instruction"]) == "instruction-0-3"


def test_frame_endpoint_prefetches_following_frames():
    visualizer = _CountingVisualizer()
    client = _make_counting_client(visualizer)

    response = client.get("/api/frames/0/0/0")

    assert response.status_code == 200
    deadline = time.time() + 1.0
    while time.time() < deadline:
        with visualizer.lock:
            if visualizer.render_count >= 3:
                break
        time.sleep(0.02)
    with visualizer.lock:
        assert visualizer.render_count >= 3


def test_frame_requests_are_stateless_for_different_episodes():
    client = _make_client()

    first = client.get("/api/frames/0/0/1")
    second = client.get("/api/frames/0/1/1")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-Step-Id"] == "2"
    assert second.headers["X-Step-Id"] == "2"


def test_concurrent_frame_requests_return_independent_step_ids():
    def get_step_id(path):
        client = _make_client()
        response = client.get(path)
        assert response.status_code == 200
        return response.headers["X-Step-Id"]

    paths = ["/api/frames/0/0/0", "/api/frames/0/0/2", "/api/frames/0/1/1"]
    with ThreadPoolExecutor(max_workers=3) as executor:
        step_ids = list(executor.map(get_step_id, paths))

    assert step_ids == ["1", "3", "2"]


def test_out_of_range_requests_return_json_errors():
    client = _make_client()

    dataset_response = client.get("/api/episodes/9/0")
    episode_response = client.get("/api/episodes/0/9")
    frame_response = client.get("/api/frames/0/0/9")

    _assert_error_response(dataset_response, 404)
    _assert_error_response(episode_response, 404)
    _assert_error_response(frame_response, 404)
