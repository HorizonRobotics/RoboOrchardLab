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

"""Dataset-level MCAP export helpers."""

from dataclasses import dataclass
from typing import Any, Mapping, cast

import fsspec

from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchEncoderConfig,
    McapBatchEncoders,
)
from robo_orchard_lab.dataset.experimental.mcap.dict2mcap import Dict2Mcap
from robo_orchard_lab.dataset.experimental.mcap.foxglove_writer import (
    FoxgloveMcapWriter as McapWriter,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.robot.columns import (
    PreservedIndexColumns,
    PreservedIndexColumnsKeys,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)

__all__ = [
    "Dataset2Mcap",
    "EpisodeInfoTopics",
]


@dataclass
class EpisodeInfoTopics:
    """A dataclass to hold the topics for episode information."""

    robot_topic: str = "/robot_description/urdf"
    """Topic for the robot description message."""
    task_topic: str = "/action/task"
    """Topic for the task message."""
    instruction_topic: str = "/action/instruction"
    """Topic for per-frame instruction messages."""
    episode_info_topic: str = "/metadata/episode"
    """Topic for one JSON message describing the exported episode."""


class Dataset2Mcap:
    """A class to save the RoboOrchard dataset to an MCAP file."""

    def __init__(self, dataset: RODataset):
        """Create an exporter for a RoboOrchard dataset."""

        self.dataset = dataset

    def save_episode(
        self,
        target_path: str,
        episode_index: int,
        encoder_cfg: Mapping[str, McapBatchEncoderConfig],
        episode_info_topics: EpisodeInfoTopics | None = None,
    ) -> None:
        """Save the episode data to an MCAP file.

        Args:
            target_path (str): The path to save the MCAP file.
            episode_index (int): The index of the episode to save.
            encoder_cfg (Mapping[str, McapBatchEncoderConfig]): The
                configuration for the MCAP batch encoder.
            episode_info_topics (EpisodeInfoTopics | None, optional):
                The topics for episode information such as robot,
                task, and instruction. If None, default topics will be used.
                Defaults to None.
        """
        if episode_info_topics is None:
            episode_info_topics = EpisodeInfoTopics()
        episode_info = self.dataset.get_meta(Episode, episode_index)
        if episode_info is None:
            raise ValueError(f"Episode with index {episode_index} not found.")
        to_mcap_msg_batch = McapBatchEncoders(encoder_cfg)
        robot_info = (
            cast(
                Robot,
                self.dataset.get_meta(Robot, episode_info.robot_index),
            )
            if episode_info.robot_index is not None
            else None
        )
        task_info = (
            cast(Task, self.dataset.get_meta(Task, episode_info.task_index))
            if episode_info.task_index is not None
            else None
        )
        begin = episode_info.dataset_begin_index
        end = begin + episode_info.frame_num

        first_frame = self.dataset.frame_dataset[begin]
        start_ts: int | None = first_frame["timestamp_min"]
        if start_ts is None:
            raise ValueError(
                f"Episode {episode_index} does not have a valid "
                "timestamp for the first frame. "
                f"To use this feature, please ensure the dataset "
                "has been properly indexed with timestamps."
            )

        dict2mcap = Dict2Mcap()

        with fsspec.open(target_path, "wb") as f, McapWriter(f) as mcap_writer:  # type: ignore
            metadata_topic_map: dict[str, list[StampedMessage[Any]]] = {
                episode_info_topics.episode_info_topic: [
                    StampedMessage(
                        data=episode_info,
                        log_time=start_ts,
                        pub_time=start_ts,
                    )
                ]
            }
            if robot_info is not None:
                metadata_topic_map[episode_info_topics.robot_topic] = [
                    StampedMessage(
                        data=robot_info,
                        log_time=start_ts,
                        pub_time=start_ts,
                    )
                ]
            if task_info is not None:
                metadata_topic_map[episode_info_topics.task_topic] = [
                    StampedMessage(
                        data=task_info,
                        log_time=start_ts,
                        pub_time=start_ts,
                    )
                ]
            dict2mcap.save_to_mcap(metadata_topic_map, mcap=mcap_writer)

            for idx in range(begin, end):
                frame = self.dataset.frame_dataset[idx]
                preserved_index_columns = PreservedIndexColumns(
                    **{k: frame.pop(k) for k in PreservedIndexColumnsKeys}
                )
                ts_min = preserved_index_columns.timestamp_min
                if ts_min is None:
                    raise ValueError(
                        f"Frame {idx} in episode {episode_index} does not have "  # noqa: E501
                        "a valid timestamp. Please ensure the dataset has been "  # noqa: E501
                        "properly indexed with timestamps."
                    )
                instruction_info = (
                    self.dataset.get_meta(
                        Instruction, preserved_index_columns.instruction_index
                    )
                    if preserved_index_columns.instruction_index is not None
                    else None
                )
                topic_map: dict[str, list[StampedMessage[Any]]] = {}
                if instruction_info is not None:
                    topic_map[episode_info_topics.instruction_topic] = [
                        StampedMessage(
                            data=instruction_info,
                            log_time=ts_min,
                            pub_time=ts_min,
                        )
                    ]

                # encode the frame data
                msg_batch = to_mcap_msg_batch.format_batch(
                    frame, raise_if_encoder_not_found=False
                )
                for topic, msgs in msg_batch.items():
                    if len(msgs) == 0:
                        continue
                    topic_map.setdefault(topic, []).extend(msgs)
                dict2mcap.save_to_mcap(topic_map, mcap=mcap_writer)
