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
import os
from typing import Any

import datasets as hg_datasets
import pytest

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    pytest.skip(
        "lerobot library not found, skipping conversion tests",
        allow_module_level=True,
    )
import torch
from PIL import Image

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    ImageMode,
)
from robo_orchard_lab.dataset.robot import (
    DatasetPackaging,
    RODataset,
)
from robo_orchard_lab.dataset.robot.conversion.lerobot_dataset import (
    LerobotDatasetEpisodePackaging,
    get_hg_features,
)

SOURCE_LEROBOT_REPO_ID = "yaak-ai/L2D-v3"
EPISODE_IDX_TO_TEST = [0]  # Test with one episode for speed
MAX_FRAMES_PER_EP = 2  # Test with 2 frames for speed


@pytest.fixture(scope="module")
def lerobot_dataset(ROBO_ORCHARD_TEST_WORKSPACE: str) -> LeRobotDataset:
    dataset = LeRobotDataset(
        SOURCE_LEROBOT_REPO_ID,
        root=os.path.join(
            ROBO_ORCHARD_TEST_WORKSPACE,
            "robo_orchard_workspace/lerobot",
            SOURCE_LEROBOT_REPO_ID,
        ),
        episodes=EPISODE_IDX_TO_TEST,
        video_backend="pyav",
    )
    return dataset


def test_lerobot_conversion_default(
    lerobot_dataset: LeRobotDataset,
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    target_path = os.path.join(tmp_path, "default_conversion")
    monkeypatch.setenv("HF_DATASETS_CACHE", os.path.join(tmp_path, ".cache"))

    target_features = get_hg_features(lerobot_dataset)

    episodes_to_package = []
    for ep_idx in EPISODE_IDX_TO_TEST:
        episode_meta = lerobot_dataset.meta.episodes[ep_idx]
        episodes_to_package.append(
            LerobotDatasetEpisodePackaging(
                lerobot_dataset,
                episode_meta,
                max_frames=MAX_FRAMES_PER_EP,
                transform=None,  # <-- Use default transform
            )
        )

    ro_dataset_packer = DatasetPackaging(
        features=target_features, check_timestamp=True
    )
    ro_dataset_packer.packaging(
        episodes=episodes_to_package,
        dataset_path=target_path,
        max_shard_size="100MB",
        force_overwrite=True,
    )

    assert os.path.exists(target_path)
    ro_dataset = RODataset(target_path)

    expected_frames = MAX_FRAMES_PER_EP * len(EPISODE_IDX_TO_TEST)
    assert len(ro_dataset) == expected_frames

    sample = ro_dataset[0]
    # Check that the image was decoded as a standard PIL Image
    camera_keys = lerobot_dataset.meta.camera_keys
    assert len(camera_keys) > 0  # Ensure the dataset has images
    for key in camera_keys:
        if key in sample:
            assert isinstance(sample[key], Image.Image)


def _get_target_features_custom(
    dataset: LeRobotDataset,
) -> hg_datasets.Features:
    features = get_hg_features(dataset)
    # Upgrade all camera data features to BatchCameraDataFeature
    for key in dataset.meta.camera_keys:
        if key in features:
            # Assumes BatchCameraData has a static method dataset_feature()
            features[key] = BatchCameraData.dataset_feature()
    return features


class _FrameDataAdaptor:
    def __init__(self, camera_keys: list[str]):
        self.camera_keys = set(camera_keys)

    def __call__(self, frame_data: dict[str, Any]) -> dict[str, Any]:
        new_frame_data = {}
        for key, value in frame_data.items():
            if key in self.camera_keys:
                # Note: Lerobot dataset will normalize the image data
                # Denormalize from [0, 1] to [0, 255]
                value_uint8 = (value * 255).type(torch.uint8)
                # Create BatchCameraData, add batch dim [B=1, ...]
                new_frame_data[key] = BatchCameraData(
                    sensor_data=value_uint8.unsqueeze(0), pix_fmt=ImageMode.RGB
                )
            else:
                # Pass other features (like 'action') through
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                new_frame_data[key] = value
        return new_frame_data


def test_lerobot_conversion_custom_transform(
    lerobot_dataset: LeRobotDataset,
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
):
    target_path = os.path.join(tmp_path, "custom_conversion")
    monkeypatch.setenv("HF_DATASETS_CACHE", os.path.join(tmp_path, ".cache"))

    # 1. Get the Custom Target Schema
    target_features = _get_target_features_custom(lerobot_dataset)

    # 2. Define the Custom Transform
    camera_keys = lerobot_dataset.meta.camera_keys
    transform_fn = _FrameDataAdaptor(camera_keys=camera_keys)

    # 3. Create Packagers (passing the custom transform)
    episodes_to_package = []
    for ep_idx in EPISODE_IDX_TO_TEST:
        episode_meta = lerobot_dataset.meta.episodes[ep_idx]
        episodes_to_package.append(
            LerobotDatasetEpisodePackaging(
                lerobot_dataset,
                episode_meta,
                max_frames=MAX_FRAMES_PER_EP,
                transform=transform_fn,  # <-- Pass the transform
            )
        )

    # 4. Run Packaging
    ro_dataset_packer = DatasetPackaging(
        features=target_features, check_timestamp=True
    )
    ro_dataset_packer.packaging(
        episodes=episodes_to_package,
        dataset_path=target_path,
        max_shard_size="100MB",
        force_overwrite=True,
    )

    # 5. Verification
    assert os.path.exists(target_path)
    ro_dataset = RODataset(target_path)

    expected_frames = MAX_FRAMES_PER_EP * len(EPISODE_IDX_TO_TEST)
    assert len(ro_dataset) == expected_frames

    sample = ro_dataset[0]

    # Core assertion: Check that the data was transformed
    # into the custom RoboOrchard data type.
    assert len(camera_keys) > 0
    for key in camera_keys:
        if key in sample:
            assert isinstance(sample[key], BatchCameraData)
            assert sample[key].pix_fmt == ImageMode.RGB
            assert sample[key].sensor_data.shape[0] == 1  # Check batch dim
