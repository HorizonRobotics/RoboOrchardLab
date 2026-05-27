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

"""Default MCAP encoder configs for RoboTwin RODatasets."""

from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchEncoderConfig,
    McapBatchFromBatchCameraDataEncodedConfig,
    McapBatchFromBatchJointStateConfig,
)

__all__ = [
    "dataset_to_mcap_config",
    "default_dataset_to_mcap_config",
]


def default_dataset_to_mcap_config() -> dict[str, McapBatchEncoderConfig]:
    """Create the default RoboTwin dataset to MCAP encoder config.

    Returns:
        dict[str, McapBatchEncoderConfig]: Encoder configs keyed by RoboTwin
            RODataset column names.
    """
    config: dict[str, McapBatchEncoderConfig] = {
        "joints": McapBatchFromBatchJointStateConfig(
            target_topic="/observation/robot_state/joints",
        ),
    }
    for camera_name in [
        "front_camera",
        "head_camera",
        "left_camera",
        "right_camera",
    ]:
        config[camera_name] = McapBatchFromBatchCameraDataEncodedConfig(
            calib_topic=f"/observation/cameras/{camera_name}/calib",
            image_topic=f"/observation/cameras/{camera_name}/image",
            tf_topic=f"/observation/cameras/{camera_name}/tf",
        )
        config[f"{camera_name}_depth"] = (
            McapBatchFromBatchCameraDataEncodedConfig(
                image_topic=f"/observation/cameras/{camera_name}/depth",
            )
        )
    return config


def dataset_to_mcap_config(
    dataset: object,
) -> dict[str, McapBatchEncoderConfig]:
    """Create the RoboTwin MCAP config for preset entry point loading.

    Args:
        dataset (object): Dataset object passed by the dataset2mcap preset
            contract. The default RoboTwin topic mapping is schema-fixed, so
            this value is not inspected.

    Returns:
        dict[str, McapBatchEncoderConfig]: Encoder configs keyed by RoboTwin
            RODataset column names.
    """
    del dataset
    return default_dataset_to_mcap_config()
