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

from __future__ import annotations
import argparse
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_DATA_FORMAT_VERSIONS = frozenset({"v1.0"})
EPISODE_FILE_RE = re.compile(r"episode_(\d+)\.hdf5$")
USD_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class EpisodeInfo:
    """Location and identity of one RoboDojo episode."""

    task_name: str
    embodiment: str
    episode_index: int
    path: Path


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _read_scalar_text(dataset: h5py.Dataset, key: str) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"{key} must be a scalar string, got {type(value)!r}.")


def _require_dataset(
    episode: h5py.File,
    key: str,
    path: Path,
) -> h5py.Dataset:
    value = episode.get(key)
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"{path}: missing required dataset '{key}'.")
    return value


def _validate_shape(
    dataset: h5py.Dataset,
    expected: tuple[int, ...],
    path: Path,
    key: str,
) -> None:
    if dataset.shape != expected:
        raise ValueError(
            f"{path}: '{key}' has shape {dataset.shape}, expected {expected}."
        )


def cam2world_usd_to_world2cam_cv(transform: np.ndarray) -> np.ndarray:
    """Convert Isaac/USD camera poses to OpenCV world-to-camera matrices."""

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            "Camera transform must end in shape (4, 4), got "
            f"{transform.shape}."
        )
    return USD_TO_OPENCV @ np.linalg.inv(transform)


def discover_episodes(
    input_path: str | Path,
    embodiment: str = "arx_x5",
    task_names: list[str] | None = None,
    max_episodes_per_task: int | None = None,
) -> list[EpisodeInfo]:
    """Discover RoboDojo episodes in deterministic task/index order."""

    root = Path(input_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"RoboDojo input directory does not exist: {root}"
        )
    if max_episodes_per_task is not None and max_episodes_per_task <= 0:
        raise ValueError("max_episodes_per_task must be positive when set.")

    requested_tasks = set(task_names) if task_names is not None else None
    episodes: list[EpisodeInfo] = []
    discovered_tasks: set[str] = set()
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_name = task_dir.name
        if requested_tasks is not None and task_name not in requested_tasks:
            continue
        data_dir = task_dir / embodiment / "data"
        if not data_dir.is_dir():
            continue

        task_episodes: list[EpisodeInfo] = []
        for episode_path in data_dir.iterdir():
            match = EPISODE_FILE_RE.fullmatch(episode_path.name)
            if match is None or not episode_path.is_file():
                continue
            task_episodes.append(
                EpisodeInfo(
                    task_name=task_name,
                    embodiment=embodiment,
                    episode_index=int(match.group(1)),
                    path=episode_path,
                )
            )
        task_episodes.sort(key=lambda episode: episode.episode_index)
        if max_episodes_per_task is not None:
            task_episodes = task_episodes[:max_episodes_per_task]
        if task_episodes:
            discovered_tasks.add(task_name)
            episodes.extend(task_episodes)

    if requested_tasks is not None:
        missing = sorted(requested_tasks - discovered_tasks)
        if missing:
            raise ValueError(
                f"No '{embodiment}' episodes found for tasks: {missing}."
            )
    if not episodes:
        raise ValueError(
            f"No RoboDojo episodes found below {root} for '{embodiment}'."
        )
    return episodes


class RoboDojoLmdbPacker(BaseLmdbManipulationDataPacker):
    """Pack RoboDojo HDF5 episodes into RoboOrchard-style LMDBs."""

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        task_names: list[str] | None = None,
        embodiment: str = "arx_x5",
        max_episodes_per_task: int | None = None,
        num_steps_per_shard: int | None = None,
        commit_step: int = 500,
        **lmdb_kwargs: Any,
    ) -> None:
        output_path = Path(output_path).expanduser().resolve()
        self._validate_output_path(output_path)
        super().__init__(
            str(input_path),
            str(output_path),
            commit_step=commit_step,
            **lmdb_kwargs,
        )
        if num_steps_per_shard is not None and num_steps_per_shard <= 0:
            raise ValueError("num_steps_per_shard must be positive when set.")
        self.embodiment = embodiment
        self.num_steps_per_shard = num_steps_per_shard
        self.episodes = discover_episodes(
            input_path,
            embodiment=embodiment,
            task_names=task_names,
            max_episodes_per_task=max_episodes_per_task,
        )
        LOGGER.info(
            "number of valid RoboDojo episodes: %d", len(self.episodes)
        )

    @staticmethod
    def _validate_output_path(output_path: Path) -> None:
        for lmdb_name in ("index", "meta", "image", "depth"):
            lmdb_path = output_path / lmdb_name
            if lmdb_path.exists() and any(lmdb_path.iterdir()):
                raise FileExistsError(
                    f"Output LMDB directory is not empty: {lmdb_path}"
                )

    @staticmethod
    def _episode_uuid(episode: EpisodeInfo) -> str:
        return (
            f"{episode.task_name}_{episode.embodiment}_"
            f"episode_{episode.episode_index:07d}"
        )

    def _load_episode(
        self,
        episode_info: EpisodeInfo,
    ) -> tuple[
        dict[str, Any],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, list[int]],
        list[str],
    ]:
        path = episode_info.path
        with h5py.File(path, "r") as episode:
            left_state = _require_dataset(
                episode, "state/left_arm_joint_states", path
            )
            if left_state.ndim != 2 or left_state.shape[1] != 6:
                raise ValueError(
                    f"{path}: 'state/left_arm_joint_states' must have shape "
                    f"(T, 6), got {left_state.shape}."
                )
            num_steps = left_state.shape[0]
            if num_steps <= 0:
                raise ValueError(
                    f"{path}: episode must contain at least one step."
                )

            expected_shapes = {
                "state/left_arm_joint_states": (num_steps, 6),
                "state/left_ee_joint_states": (num_steps, 1),
                "state/right_arm_joint_states": (num_steps, 6),
                "state/right_ee_joint_states": (num_steps, 1),
                "action/left_arm_joint_states": (num_steps, 6),
                "action/left_ee_joint_states": (num_steps, 1),
                "action/right_arm_joint_states": (num_steps, 6),
                "action/right_ee_joint_states": (num_steps, 1),
                "state/left_ee_poses": (num_steps, 7),
                "state/right_ee_poses": (num_steps, 7),
                "state/left_delta_ee_poses": (num_steps, 7),
                "state/right_delta_ee_poses": (num_steps, 7),
            }
            datasets = {}
            for key, expected_shape in expected_shapes.items():
                dataset = _require_dataset(episode, key, path)
                _validate_shape(dataset, expected_shape, path, key)
                datasets[key] = dataset

            version_dataset = _require_dataset(
                episode, "data_format_version", path
            )
            version = _read_scalar_text(version_dataset, "data_format_version")
            if version not in SUPPORTED_DATA_FORMAT_VERSIONS:
                raise ValueError(
                    f"{path}: unsupported data_format_version {version!r}."
                )
            instruction = _read_scalar_text(
                _require_dataset(episode, "instruction", path), "instruction"
            )
            frequency_dataset = _require_dataset(
                episode, "additional_info/frequency", path
            )
            frequency = int(frequency_dataset[()])
            if frequency <= 0:
                raise ValueError(f"{path}: frequency must be positive.")

            joint_positions = np.concatenate(
                [
                    datasets["state/left_arm_joint_states"][:],
                    datasets["state/left_ee_joint_states"][:],
                    datasets["state/right_arm_joint_states"][:],
                    datasets["state/right_ee_joint_states"][:],
                ],
                axis=1,
            )
            master_joint_positions = np.concatenate(
                [
                    datasets["action/left_arm_joint_states"][:],
                    datasets["action/left_ee_joint_states"][:],
                    datasets["action/right_arm_joint_states"][:],
                    datasets["action/right_ee_joint_states"][:],
                ],
                axis=1,
            )
            cartesian_position = np.stack(
                [
                    datasets["state/left_ee_poses"][:],
                    datasets["state/right_ee_poses"][:],
                ],
                axis=1,
            )
            delta_cartesian_position = np.stack(
                [
                    datasets["state/left_delta_ee_poses"][:],
                    datasets["state/right_delta_ee_poses"][:],
                ],
                axis=1,
            )

            intrinsics: dict[str, np.ndarray] = {}
            extrinsics: dict[str, np.ndarray] = {}
            camera_shapes: dict[str, list[int]] = {}
            vision = episode.get("vision")
            if not isinstance(vision, h5py.Group):
                raise ValueError(f"{path}: missing required group 'vision'.")
            camera_names = sorted(
                name
                for name, value in vision.items()
                if isinstance(value, h5py.Group)
            )
            if not camera_names:
                raise ValueError(f"{path}: 'vision' does not contain cameras.")
            for camera_name in camera_names:
                prefix = f"vision/{camera_name}"
                colors = _require_dataset(episode, f"{prefix}/colors", path)
                _validate_shape(colors, (num_steps,), path, f"{prefix}/colors")
                intrinsic = _require_dataset(
                    episode, f"{prefix}/intrinsic_matrix", path
                )
                _validate_shape(
                    intrinsic, (3, 3), path, f"{prefix}/intrinsic_matrix"
                )
                extrinsic = _require_dataset(
                    episode, f"{prefix}/extrinsic_matrix", path
                )
                _validate_shape(
                    extrinsic,
                    (num_steps, 4, 4),
                    path,
                    f"{prefix}/extrinsic_matrix",
                )
                shape = _require_dataset(episode, f"{prefix}/shape", path)
                _validate_shape(shape, (3,), path, f"{prefix}/shape")
                intrinsics[camera_name] = np.asarray(
                    intrinsic[:], dtype=np.float64
                )
                try:
                    extrinsics[camera_name] = cam2world_usd_to_world2cam_cv(
                        extrinsic[:]
                    )
                except np.linalg.LinAlgError as error:
                    raise ValueError(
                        f"{path}: non-invertible extrinsic for "
                        f"'{camera_name}'."
                    ) from error
                camera_shapes[camera_name] = [int(value) for value in shape[:]]

            arrays = {
                "observation/robot_state/joint_positions": joint_positions,
                "observation/robot_state/master_joint_positions": (
                    master_joint_positions
                ),
                "observation/robot_state/cartesian_position": (
                    cartesian_position
                ),
                "observation/robot_state/delta_cartesian_position": (
                    delta_cartesian_position
                ),
            }
            episode_meta = {
                "task_name": episode_info.task_name,
                "embodiment": episode_info.embodiment,
                "episode_index": episode_info.episode_index,
                "num_steps": num_steps,
                "frequency": frequency,
                "data_format_version": version,
                "instruction": instruction,
                "simulation": True,
            }
        return (
            episode_meta,
            arrays,
            intrinsics,
            extrinsics,
            camera_shapes,
            camera_names,
        )

    def _write_images(
        self,
        uuid: str,
        episode_info: EpisodeInfo,
        camera_names: list[str],
    ) -> None:
        with h5py.File(episode_info.path, "r") as episode:
            for camera_name in camera_names:
                colors = episode[f"vision/{camera_name}/colors"]
                for step_index in range(len(colors)):
                    encoded = bytes(colors[step_index])
                    if not (
                        encoded.startswith(b"\xff\xd8")
                        and encoded.endswith(b"\xff\xd9")
                    ):
                        raise ValueError(
                            f"{episode_info.path}: invalid JPEG at "
                            f"{camera_name}[{step_index}]."
                        )
                    self.image_pack_file.write(
                        f"{uuid}/{camera_name}/{step_index}", encoded
                    )

    def _write_timeseries(
        self,
        uuid: str,
        arrays: dict[str, np.ndarray],
    ) -> None:
        num_steps = len(next(iter(arrays.values())))
        if self.num_steps_per_shard is None:
            for key, value in arrays.items():
                self.meta_pack_file.write(f"{uuid}/{key}", value)
            return

        self.meta_pack_file.write(
            f"{uuid}/num_steps_per_shard", self.num_steps_per_shard
        )
        num_shards = math.ceil(num_steps / self.num_steps_per_shard)
        for shard_index in range(num_shards):
            start = shard_index * self.num_steps_per_shard
            end = min(start + self.num_steps_per_shard, num_steps)
            for key, value in arrays.items():
                self.meta_pack_file.write(
                    f"{uuid}/{shard_index}/{key}", value[start:end]
                )

    def _pack_episode(
        self, output_index: int, episode: EpisodeInfo
    ) -> dict[str, Any]:
        uuid = self._episode_uuid(episode)
        LOGGER.info(
            "start process [%d/%d] %s",
            output_index + 1,
            len(self.episodes),
            uuid,
        )
        (
            metadata,
            arrays,
            intrinsics,
            extrinsics,
            camera_shapes,
            camera_names,
        ) = self._load_episode(episode)
        self._write_images(uuid, episode, camera_names)

        index_data = {"uuid": uuid, **metadata}
        self.meta_pack_file.write(f"{uuid}/meta_data", index_data)
        self.meta_pack_file.write(
            f"{uuid}/instructions", metadata["instruction"]
        )
        self.meta_pack_file.write(f"{uuid}/camera_names", camera_names)
        self.meta_pack_file.write(f"{uuid}/camera_shapes", camera_shapes)
        self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsics)
        self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsics)
        self._write_timeseries(uuid, arrays)
        LOGGER.info(
            "finish process %s, num_steps:%d", uuid, metadata["num_steps"]
        )
        return index_data

    def _pack(self) -> None:
        try:
            index_records = []
            for output_index, episode in enumerate(self.episodes):
                index_records.append(self._pack_episode(output_index, episode))
            for output_index, index_data in enumerate(index_records):
                self.write_index(output_index, index_data)
            self.index_pack_file.write(
                "__len__", len(self.episodes), commit=True
            )
            self.image_pack_file.close()
            self.depth_pack_file.close()
            self.index_pack_file.close()
            self.meta_pack_file.write("__pack_complete__", True, commit=True)
            self.meta_pack_file.close()
        finally:
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack RoboDojo HDF5 data into RoboOrchard-style LMDB."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--task_names",
        default=None,
        help="Optional comma-separated task names.",
    )
    parser.add_argument("--embodiment", default="arx_x5")
    parser.add_argument("--max_episodes_per_task", type=int, default=None)
    parser.add_argument("--num_steps_per_shard", type=int, default=None)
    parser.add_argument("--commit_step", type=int, default=500)
    parser.add_argument("--map_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
    )
    args = parse_args()
    lmdb_kwargs = {}
    if args.map_size is not None:
        lmdb_kwargs["map_size"] = args.map_size
    packer = RoboDojoLmdbPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        task_names=_parse_csv(args.task_names),
        embodiment=args.embodiment,
        max_episodes_per_task=args.max_episodes_per_task,
        num_steps_per_shard=args.num_steps_per_shard,
        commit_step=args.commit_step,
        **lmdb_kwargs,
    )
    packer()


if __name__ == "__main__":
    main()
