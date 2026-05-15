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

# ruff: noqa: E402, I001

import argparse
import copy
import html
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from math import ceil
from typing import Any
from urllib.parse import quote

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_COMMON_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(
    os.path.join(_COMMON_DIR, os.pardir, os.pardir, os.pardir)
)
_INDEX_HTML_PATH = os.path.join(_THIS_DIR, "index.html")
for _path in (_COMMON_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from holobrain_utils import (  # noqa: E402
    HolobrainDataFeature,
    HolobrainVideoVisualizer,
    load_config,
)

logger = logging.getLogger(__file__)
EncodedFrame = tuple[bytes, int, str, str, int]


@dataclass(slots=True)
class DatasetHandle:
    """Server-side state for one selectable visualization dataset."""

    dataset_id: int
    name: str
    dataset: Any | None = None
    visualizer: HolobrainVideoVisualizer | None = None
    datasets: dict[str, Any] = field(default_factory=dict)
    visualizers: dict[str, HolobrainVideoVisualizer] = field(
        default_factory=dict
    )
    config: dict[str, Any] | None = None
    build_training_dataset: Any | None = None
    build_validation_dataset: Any | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    build_lock: threading.Lock = field(default_factory=threading.Lock)


class FrameCache:
    """Small thread-safe LRU cache for encoded frame responses."""

    def __init__(self, max_items: int):
        self.max_items = max(0, max_items)
        self._items: OrderedDict[tuple[Any, ...], EncodedFrame] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> EncodedFrame | None:
        if self.max_items <= 0:
            return None
        with self._lock:
            cached = self._items.get(key)
            if cached is None:
                return None
            self._items.move_to_end(key)
            return cached

    def put(self, key: tuple[Any, ...], value: EncodedFrame) -> None:
        if self.max_items <= 0:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)


class FramePrefetcher:
    """Background prefetch scheduler for sequential frame playback."""

    def __init__(self, max_workers: int):
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._pending: set[tuple[Any, ...]] = set()
        self._lock = threading.Lock()

    def submit(
        self,
        key: tuple[Any, ...],
        fn,
        *args,
    ) -> None:
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
        future = self._executor.submit(fn, *args)
        future.add_done_callback(lambda _future: self._mark_done(key))

    def _mark_done(self, key: tuple[Any, ...]) -> None:
        with self._lock:
            self._pending.discard(key)


class EpisodeConcatDataset:
    """Episode-addressable view over multiple frame-addressable datasets."""

    def __init__(self, name: str, datasets: list[Any]):
        self.dataset_name = name
        self.datasets = datasets
        frame_counts = [len(dataset) for dataset in datasets]
        episode_counts = [_episode_count(dataset) for dataset in datasets]
        self._frame_offsets = np.cumsum([0] + frame_counts)
        self._episode_offsets = np.cumsum([0] + episode_counts)
        self.num_episode = int(self._episode_offsets[-1])

    def __len__(self) -> int:
        return int(self._frame_offsets[-1])

    def __getitem__(self, index: int):
        dataset_idx = int(
            np.searchsorted(self._frame_offsets, index, side="right") - 1
        )
        local_index = index - int(self._frame_offsets[dataset_idx])
        return self.datasets[dataset_idx][local_index]

    def get_episode_range(self, ep_idx: int) -> tuple[int, int]:
        if ep_idx < 0 or ep_idx >= self.num_episode:
            raise IndexError(f"Episode id {ep_idx} is out of range.")
        dataset_idx = int(
            np.searchsorted(self._episode_offsets, ep_idx, side="right") - 1
        )
        local_ep_idx = ep_idx - int(self._episode_offsets[dataset_idx])
        local_start, local_end = self.datasets[dataset_idx].get_episode_range(
            local_ep_idx
        )
        frame_offset = int(self._frame_offsets[dataset_idx])
        return frame_offset + int(local_start), frame_offset + int(local_end)


class EpisodeIndexDataset:
    """Episode-addressable wrapper for frame datasets with episode_index."""

    def __init__(self, dataset: Any):
        self.dataset = dataset
        self.dataset_name = getattr(dataset, "dataset_name", None)
        self._episode_ranges = self._build_episode_ranges()
        self.num_episode = len(self._episode_ranges)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def get_episode_range(self, ep_idx: int) -> tuple[int, int]:
        if ep_idx < 0 or ep_idx >= self.num_episode:
            raise IndexError(f"Episode id {ep_idx} is out of range.")
        return self._episode_ranges[ep_idx]

    def _build_episode_ranges(self) -> list[tuple[int, int]]:
        episode_indices = _episode_indices(self.dataset)
        ranges = []
        start = 0
        current = episode_indices[0] if episode_indices else None
        for index, episode_index in enumerate(episode_indices):
            if episode_index == current:
                continue
            ranges.append((start, index))
            start = index
            current = episode_index
        if episode_indices:
            ranges.append((start, len(episode_indices)))
        return ranges


def build_dataset_handles(
    config_path: str,
    dataset_names: list[str] | None,
) -> list[DatasetHandle]:
    """Create lazy selectable dataset handles from the visualizer config."""
    config_path = resolve_config_path(config_path)
    config = load_config(config_path)
    cfg = copy.deepcopy(config.config)
    training_names = list(cfg.get("training_datasets", []))
    validation_names = list(cfg.get("validation_datasets", []))
    if dataset_names:
        names = list(dict.fromkeys(dataset_names))
    else:
        names = sorted(
            set(training_names)
            | set(validation_names)
            | set(cfg.get("deploy_datasets", []))
        )
    return [
        DatasetHandle(
            dataset_id=dataset_id,
            name=name,
            config=cfg,
            build_training_dataset=config.build_training_dataset,
            build_validation_dataset=config.build_validation_dataset,
            lock=threading.Lock(),
            build_lock=threading.Lock(),
        )
        for dataset_id, name in enumerate(names)
    ]


def create_app(
    dataset_handles: list[DatasetHandle],
    fps: int,
    interval: int,
    max_cache_frames: int = 256,
    prefetch_frames: int = 8,
    prefetch_workers: int = 2,
) -> Flask:
    """Create the HoloBrain visualization web app."""
    app = Flask(__name__)
    handles = {handle.dataset_id: handle for handle in dataset_handles}
    frame_cache = FrameCache(max_cache_frames)
    prefetcher = FramePrefetcher(prefetch_workers)

    def load_request_dataset(
        dataset_id: int,
    ) -> tuple[DatasetHandle, str, int]:
        handle = handles.get(dataset_id)
        if handle is None:
            raise UnknownDatasetError("Unknown dataset id.")
        mode = _request_mode()
        vis_interval = _request_vis_interval(interval)
        _ensure_dataset_built(handle, mode)
        return handle, mode, vis_interval

    @app.route("/")
    def index() -> Response:
        dataset_payloads = _dataset_payloads(handles.values())
        html = (
            _index_html()
            .replace(
                "__INITIAL_DATASETS__",
                json.dumps(dataset_payloads, ensure_ascii=False),
            )
            .replace(
                "__DATASET_OPTIONS__",
                _dataset_option_html(dataset_payloads),
            )
        )
        response = Response(html, mimetype="text/html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/datasets")
    def list_datasets():
        response = jsonify({"datasets": _dataset_payloads(handles.values())})
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/episodes/<int:dataset_id>/<int:episode_id>")
    def get_episode(dataset_id: int, episode_id: int):
        try:
            handle, mode, vis_interval = load_request_dataset(dataset_id)
            metadata = _episode_metadata(
                handle, mode, episode_id, fps, vis_interval
            )
        except (IndexError, UnknownDatasetError) as exc:
            return _json_error(str(exc), 404)
        except (UnsupportedDatasetModeError, ValueError) as exc:
            return _json_error(str(exc), 400)
        except Exception as exc:
            logger.exception("Failed to load episode metadata")
            return _json_error(str(exc), 500)
        return jsonify(metadata)

    @app.route(
        "/api/frames/<int:dataset_id>/<int:episode_id>/<int:frame_offset>"
    )
    def get_frame(dataset_id: int, episode_id: int, frame_offset: int):
        try:
            handle, mode, vis_interval = load_request_dataset(dataset_id)
            highlight_joint_indices = _request_highlight_joint_indices()
            cached = _cached_or_render_frame(
                frame_cache,
                (
                    dataset_id,
                    mode,
                    episode_id,
                    frame_offset,
                    vis_interval,
                    highlight_joint_indices,
                ),
                handle,
                mode,
                episode_id,
                frame_offset,
                vis_interval,
                highlight_joint_indices,
            )
        except (IndexError, UnknownDatasetError) as exc:
            return _json_error(str(exc), 404)
        except (UnsupportedDatasetModeError, ValueError) as exc:
            return _json_error(str(exc), 400)
        except Exception as exc:
            logger.exception("Failed to render frame")
            return _json_error(str(exc), 500)
        _schedule_prefetch(
            prefetcher,
            frame_cache,
            handle,
            dataset_id,
            mode,
            episode_id,
            frame_offset,
            vis_interval,
            highlight_joint_indices,
            prefetch_frames,
        )

        return _frame_response(cached)

    @app.route("/api/frames/<int:dataset_id>/<int:episode_id>/locate")
    def locate_frame(dataset_id: int, episode_id: int):
        try:
            handle, mode, vis_interval = load_request_dataset(dataset_id)
            data_index = _request_data_index()
            dataset = _loaded_dataset(handle, mode)
            assert dataset is not None
            start, end = _episode_range(dataset, episode_id)
            frame_offset = _frame_offset_for_dataset_index(
                start, end, data_index, vis_interval
            )
        except (IndexError, UnknownDatasetError) as exc:
            return _json_error(str(exc), 404)
        except (UnsupportedDatasetModeError, ValueError) as exc:
            return _json_error(str(exc), 400)
        except Exception as exc:
            logger.exception("Failed to locate frame")
            return _json_error(str(exc), 500)
        response = jsonify(
            {
                "dataset_id": dataset_id,
                "mode": mode,
                "episode_id": episode_id,
                "data_index": data_index,
                "frame_offset": frame_offset,
            }
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    return app


def _index_html() -> str:
    with open(_INDEX_HTML_PATH, encoding="utf-8") as file:
        return file.read()


def _dataset_payloads(handles) -> list[dict[str, Any]]:
    return [
        {
            "id": handle.dataset_id,
            "name": handle.name,
            "loaded_modes": sorted(handle.datasets.keys()),
            "num_episodes": {
                mode: _episode_count(dataset)
                for mode, dataset in handle.datasets.items()
            },
        }
        for handle in handles
    ]


def _dataset_option_html(dataset_payloads: list[dict[str, Any]]) -> str:
    options = []
    for item in dataset_payloads:
        options.append(
            '<option value="{}">{}</option>'.format(
                html.escape(str(item["id"]), quote=True),
                html.escape(_dataset_label_text(item), quote=False),
            )
        )
    return "\n".join(options)


def _dataset_label_text(item: dict[str, Any]) -> str:
    loaded_modes = item.get("loaded_modes") or []
    if len(loaded_modes) == 0:
        loaded_text = "not loaded"
    else:
        loaded_text = "loaded: " + "/".join(str(x) for x in loaded_modes)
    return f"{item['name']} ({loaded_text})"


def _schedule_prefetch(
    prefetcher: FramePrefetcher,
    frame_cache: FrameCache,
    handle: DatasetHandle,
    dataset_id: int,
    mode: str,
    episode_id: int,
    frame_offset: int,
    interval: int,
    highlight_joint_indices: tuple[int, ...],
    prefetch_frames: int,
) -> None:
    dataset = _loaded_dataset(handle, mode)
    if prefetch_frames <= 0 or dataset is None:
        return
    try:
        start, end = _episode_range(dataset, episode_id)
    except IndexError:
        return
    max_offset = max(0, _frame_count(start, end, interval) - 1)
    for offset in range(
        frame_offset + 1,
        min(max_offset, frame_offset + prefetch_frames) + 1,
    ):
        key = (
            dataset_id,
            mode,
            episode_id,
            offset,
            interval,
            highlight_joint_indices,
            "",
        )
        if frame_cache.get(key) is not None:
            continue
        prefetcher.submit(
            key,
            _prefetch_frame,
            frame_cache,
            key,
            handle,
            mode,
            episode_id,
            offset,
            interval,
            highlight_joint_indices,
        )


def _prefetch_frame(
    frame_cache: FrameCache,
    key: tuple[Any, ...],
    handle: DatasetHandle,
    mode: str,
    episode_id: int,
    frame_offset: int,
    interval: int,
    highlight_joint_indices: tuple[int, ...],
) -> None:
    if frame_cache.get(key) is not None:
        return
    try:
        frame_cache.put(
            key,
            _render_encoded_frame(
                handle,
                mode,
                episode_id,
                frame_offset,
                interval,
                highlight_joint_indices,
            ),
        )
    except Exception:
        logger.debug("Frame prefetch failed for key %s", key, exc_info=True)


def resolve_config_path(config_path: str) -> str:
    """Resolve config paths before the app changes cwd to the common dir."""
    if os.path.isabs(config_path):
        return config_path
    candidates = [
        os.path.abspath(config_path),
        os.path.join(_REPO_ROOT, config_path),
        os.path.join(_COMMON_DIR, config_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


class UnsupportedDatasetModeError(RuntimeError):
    """Raised when a dataset name is not available for the requested mode."""


class UnknownDatasetError(LookupError):
    """Raised when a request references an unknown dataset id."""


def _request_mode() -> str:
    mode = request.args.get("mode", "training")
    if mode not in {"training", "validation"}:
        raise UnsupportedDatasetModeError(f"Unsupported mode: {mode}.")
    return mode


def _request_vis_interval(default_interval: int) -> int:
    raw_value = request.args.get("vis_interval")
    if raw_value is None:
        return default_interval
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("vis_interval must be an integer.") from exc
    if value < 1:
        raise ValueError("vis_interval must be >= 1.")
    return value


def _request_highlight_joint_indices() -> tuple[int, ...]:
    raw_value = request.args.get("highlight_joint_indices", "6,13")
    tokens = [token.strip() for token in raw_value.split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError(
            "highlight_joint_indices must contain at least one integer."
        )
    try:
        return tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(
            "highlight_joint_indices must be a comma-separated list of "
            "integers."
        ) from exc


def _request_data_index() -> int:
    raw_value = request.args.get("data_index")
    if raw_value is None:
        raise ValueError("data_index is required.")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError("data_index must be an integer.") from exc


def _ensure_dataset_built(handle: DatasetHandle, mode: str) -> None:
    if mode in handle.datasets:
        return
    with handle.build_lock:
        if mode in handle.datasets:
            return
        build_dataset, config_key = _mode_builder(handle, mode)
        if build_dataset is None:
            raise UnsupportedDatasetModeError(
                f"Dataset {handle.name} does not support {mode} mode."
            )
        if handle.config is None:
            raise RuntimeError(f"Dataset {handle.name} is not configured.")
        cfg = copy.deepcopy(handle.config)
        cfg[config_key] = [handle.name]
        start_time = time.monotonic()
        logger.info("Building dataset %s in %s mode", handle.name, mode)
        concat_dataset = build_dataset(cfg)
        if concat_dataset is None:
            raise UnsupportedDatasetModeError(
                f"Dataset {handle.name} does not support {mode} mode."
            )
        datasets = _episode_datasets(concat_dataset)
        if len(datasets) == 0:
            raise UnsupportedDatasetModeError(
                f"Dataset {handle.name} does not support {mode} mode."
            )
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = EpisodeConcatDataset(handle.name, datasets)
        handle.datasets[mode] = dataset
        handle.visualizers[mode] = HolobrainVideoVisualizer(dataset)
        if handle.dataset is None:
            handle.dataset = dataset
            handle.visualizer = handle.visualizers[mode]
        logger.info(
            "Built dataset %s in %s mode with %s episodes in %.1fs",
            handle.name,
            mode,
            _episode_count(dataset),
            time.monotonic() - start_time,
        )


def _episode_datasets(dataset: Any) -> list[Any]:
    if _has_episode_range(dataset):
        return [_ensure_episode_dataset(dataset)]
    sub_datasets = getattr(dataset, "datasets", None)
    if sub_datasets is None:
        return [_ensure_episode_dataset(dataset)]
    datasets = []
    for sub_dataset in sub_datasets:
        datasets.extend(_episode_datasets(sub_dataset))
    return datasets


def _ensure_episode_dataset(dataset: Any) -> Any:
    if _has_episode_range(dataset):
        return dataset
    return EpisodeIndexDataset(dataset)


def _has_episode_range(dataset: Any) -> bool:
    return callable(getattr(dataset, "get_episode_range", None))


def _mode_builder(handle: DatasetHandle, mode: str) -> tuple[Any, str]:
    if mode == "training":
        return handle.build_training_dataset, "training_datasets"
    if mode == "validation":
        return handle.build_validation_dataset, "validation_datasets"
    raise UnsupportedDatasetModeError(f"Unsupported mode: {mode}.")


def _loaded_dataset(handle: DatasetHandle, mode: str) -> Any | None:
    if mode in handle.datasets:
        return handle.datasets[mode]
    if mode == "training" and not handle.datasets:
        return handle.dataset
    return None


def _episode_count(dataset: Any) -> int:
    for name in (
        "num_episode",
        "episode_num",
        "cumsum_steps",
        "episode_indices",
        "episodes",
        "idx_lmdbs",
    ):
        value = getattr(dataset, name, None)
        if value is not None:
            break
    if callable(value):
        value = value()
    if value is None:
        return _count_episode_ranges(dataset)
    try:
        return int(value)
    except (TypeError, ValueError):
        return len(value)


def _episode_indices(dataset: Any) -> list[Any]:
    for source in (
        getattr(dataset, "index_dataset", None),
        getattr(dataset, "frame_dataset", None),
        dataset,
    ):
        if source is None:
            continue
        try:
            values = source["episode_index"]
        except Exception:
            continue
        return list(values)
    values = [
        _frame_episode_index(dataset[index]) for index in range(len(dataset))
    ]
    return list(values)


def _frame_episode_index(frame: dict[str, Any]) -> Any:
    if "episode_index" not in frame:
        raise AttributeError("Dataset does not expose episode_index.")
    return frame["episode_index"]


def _count_episode_ranges(dataset: Any) -> int:
    get_episode_range = getattr(dataset, "get_episode_range", None)
    if get_episode_range is None:
        raise AttributeError("Dataset does not expose episode count.")
    max_checks = max(1, len(dataset) + 1)
    for episode_id in range(max_checks):
        try:
            get_episode_range(episode_id)
        except IndexError:
            return episode_id
    raise AttributeError("Dataset episode count could not be inferred.")


def _episode_range(dataset: Any, episode_id: int) -> tuple[int, int]:
    if episode_id < 0 or episode_id >= _episode_count(dataset):
        raise IndexError(f"Episode id {episode_id} is out of range.")
    start, end = dataset.get_episode_range(episode_id)
    return int(start), int(end)


def _frame_count(start: int, end: int, interval: int) -> int:
    return max(0, ceil((end - start - 1) / interval))


def _dataset_index(
    start: int, end: int, frame_offset: int, interval: int
) -> int:
    if frame_offset < 0:
        raise IndexError(f"Frame offset {frame_offset} is out of range.")
    dataset_index = start + _episode_step_id(frame_offset, interval)
    if dataset_index >= end:
        raise IndexError(f"Frame offset {frame_offset} is out of range.")
    return dataset_index


def _frame_offset_for_dataset_index(
    start: int, end: int, data_index: int, interval: int
) -> int:
    if data_index <= start or data_index >= end:
        raise IndexError(f"Data index {data_index} is out of episode range.")
    offset, remainder = divmod(data_index - start - 1, interval)
    if remainder != 0:
        raise ValueError(
            f"Data index {data_index} is not aligned to vis_interval "
            f"{interval}."
        )
    return offset


def _episode_metadata(
    handle: DatasetHandle,
    mode: str,
    episode_id: int,
    fps: int,
    interval: int,
) -> dict[str, Any]:
    dataset = _loaded_dataset(handle, mode)
    assert dataset is not None
    start, end = _episode_range(dataset, episode_id)
    if end <= start:
        raise IndexError(f"Episode id {episode_id} is empty.")
    num_frames = _frame_count(start, end, interval)
    first_index = start + 1 if num_frames > 0 else start
    with handle.lock:
        first_frame = dataset[first_index]
    return {
        "dataset_id": handle.dataset_id,
        "dataset_name": handle.name,
        "mode": mode,
        "episode_id": episode_id,
        "uuid": str(first_frame.get("uuid", "")),
        "instruction": _string_value(first_frame.get("text", "")),
        "subtask": _string_value(
            first_frame.get("subtask", first_frame.get("subtask_text", ""))
        ),
        "num_frames": num_frames,
        "num_episodes": _episode_count(dataset),
        "fps": fps,
        "interval": interval,
        "first_step_id": _episode_step_id(0, interval),
    }


def _render_encoded_frame(
    handle: DatasetHandle,
    mode: str,
    episode_id: int,
    frame_offset: int,
    interval: int,
    highlight_joint_indices: tuple[int, ...],
) -> EncodedFrame:
    dataset = _loaded_dataset(handle, mode)
    visualizer = handle.visualizers.get(mode, handle.visualizer)
    assert dataset is not None
    assert visualizer is not None
    start, end = _episode_range(dataset, episode_id)
    dataset_index = _dataset_index(start, end, frame_offset, interval)

    with handle.lock:
        raw_frame = dataset[dataset_index]
    rendered = visualizer._render_frame(
        HolobrainDataFeature.from_dict(raw_frame),
        ee_indices=highlight_joint_indices,
    )
    return (
        _encode_jpeg(rendered),
        _episode_step_id(frame_offset, interval),
        _string_value(raw_frame.get("text", "")),
        _string_value(
            raw_frame.get("subtask", raw_frame.get("subtask_text", ""))
        ),
        dataset_index,
    )


def _cached_or_render_frame(
    frame_cache: FrameCache,
    cache_key: tuple[Any, ...],
    handle: DatasetHandle,
    mode: str,
    episode_id: int,
    frame_offset: int,
    interval: int,
    highlight_joint_indices: tuple[int, ...],
) -> EncodedFrame:
    cached = frame_cache.get(cache_key)
    if cached is not None:
        return cached
    rendered = _render_encoded_frame(
        handle,
        mode,
        episode_id,
        frame_offset,
        interval,
        highlight_joint_indices,
    )
    frame_cache.put(cache_key, rendered)
    return rendered


def _frame_response(frame: EncodedFrame) -> Response:
    payload, step_id, instruction, subtask, data_index = frame
    response = Response(payload, mimetype="image/jpeg")
    response.headers["X-Step-Id"] = str(step_id)
    response.headers["X-Instruction"] = quote(instruction)
    response.headers["X-Subtask"] = quote(subtask)
    response.headers["X-Data-Index"] = str(data_index)
    response.headers["Cache-Control"] = "no-store"
    return response


def _episode_step_id(frame_offset: int, interval: int) -> int:
    return 1 + frame_offset * interval


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def _encode_jpeg(frame: np.ndarray) -> bytes:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    image = Image.fromarray(frame)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _json_error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_names", type=str, nargs="+")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--max_cache_frames", type=int, default=256)
    parser.add_argument("--prefetch_frames", type=int, default=8)
    parser.add_argument("--prefetch_workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    from gevent.pywsgi import WSGIServer

    args = parse_args()
    config_path = resolve_config_path(args.config)
    os.chdir(_COMMON_DIR)
    logging.basicConfig(
        format=(
            "%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s"
        ),
        level=logging.INFO,
    )
    dataset_handles = build_dataset_handles(config_path, args.dataset_names)
    app = create_app(
        dataset_handles=dataset_handles,
        fps=args.fps,
        interval=args.interval,
        max_cache_frames=args.max_cache_frames,
        prefetch_frames=args.prefetch_frames,
        prefetch_workers=args.prefetch_workers,
    )
    logger.info(
        "Serving HoloBrain data visualizer at http://%s:%s",
        args.host,
        args.port,
    )
    WSGIServer((args.host, args.port), app).serve_forever()


if __name__ == "__main__":
    main()
