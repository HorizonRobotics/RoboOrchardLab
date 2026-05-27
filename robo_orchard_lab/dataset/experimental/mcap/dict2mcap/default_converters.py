# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

"""Default Dict2Mcap converters for RoboOrchard dataset message types."""

from __future__ import annotations
import os
from typing import Any, Generic, TypeVar

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchFrameTransform,
    BatchFrameTransformGraph,
    BatchJointsState,
    BatchPose,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchFromBatchCameraDataConfig,
    McapBatchFromBatchCameraDataEncodedConfig,
    McapBatchFromBatchFrameTransformConfig,
    McapBatchFromBatchFrameTransformGraphConfig,
    McapBatchFromBatchJointStateConfig,
    McapBatchFromBatchPoseConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.dict2mcap.converters import (
    BatchFormatConverter,
    MessageDict,
    ToMcapMessageFactory,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    FromEpisodeConfig,
    FromInstructionConfig,
    FromRobotConfig,
    FromTaskConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.topics import (
    camera_image_topic,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)

__all__ = ["create_default_to_mcap_message_factory"]

T = TypeVar("T")


def create_default_to_mcap_message_factory() -> ToMcapMessageFactory:
    """Create default converters for RoboOrchard batch and ORM objects.

    The returned factory covers common dataset batch datatypes plus
    ``Robot``, ``Task``, ``Instruction``, and ``Episode`` ORM rows. ORM
    converters preserve the source topic and stamped message times while
    converting storage rows into Pydantic metadata payloads.
    """

    factory = ToMcapMessageFactory()

    @factory.register(BatchJointsState)
    def _default_joint_state_converter(
        topic: str, data: BatchJointsState, **kwargs: Any
    ) -> BatchFormatConverter[BatchJointsState]:
        return BatchFormatConverter(
            McapBatchFromBatchJointStateConfig(
                target_topic=topic
            )().format_batch
        )

    @factory.register(BatchPose)
    def _default_pose_converter(
        topic: str, data: BatchPose, **kwargs: Any
    ) -> BatchFormatConverter[BatchPose]:
        return BatchFormatConverter(
            McapBatchFromBatchPoseConfig(target_topic=topic)().format_batch
        )

    @factory.register(BatchFrameTransform)
    def _default_frame_transform_converter(
        topic: str, data: BatchFrameTransform, **kwargs: Any
    ) -> BatchFormatConverter[BatchFrameTransform]:
        return BatchFormatConverter(
            McapBatchFromBatchFrameTransformConfig(
                target_topic=topic
            )().format_batch
        )

    @factory.register(BatchFrameTransformGraph)
    def _default_frame_transform_graph_converter(
        topic: str, data: BatchFrameTransformGraph, **kwargs: Any
    ) -> BatchFormatConverter[BatchFrameTransformGraph]:
        return BatchFormatConverter(
            McapBatchFromBatchFrameTransformGraphConfig(
                target_topic=topic
            )().format_batch
        )

    @factory.register(BatchCameraDataEncoded)
    def _default_camera_data_encoded_converter(
        topic: str, data: BatchCameraDataEncoded, **kwargs: Any
    ) -> BatchFormatConverter[BatchCameraDataEncoded]:
        return BatchFormatConverter(
            McapBatchFromBatchCameraDataEncodedConfig(
                image_topic=camera_image_topic(topic),
                calib_topic=os.path.join(topic, "calib"),
                tf_topic=os.path.join(topic, "tf"),
            )().format_batch
        )

    @factory.register(BatchCameraData)
    def _default_camera_data_converter(
        topic: str, data: BatchCameraData, **kwargs: Any
    ) -> BatchFormatConverter[BatchCameraData]:
        return BatchFormatConverter(
            McapBatchFromBatchCameraDataConfig(
                image_topic=camera_image_topic(topic),
                calib_topic=os.path.join(topic, "calib"),
                tf_topic=os.path.join(topic, "tf"),
            )().format_batch
        )

    @factory.register(Robot)
    def _default_robot_converter(
        topic: str, data: Robot, **kwargs: Any
    ) -> _OrmConverter[Robot]:
        return _OrmConverter(topic, FromRobotConfig()())

    @factory.register(Task)
    def _default_task_converter(
        topic: str, data: Task, **kwargs: Any
    ) -> _OrmConverter[Task]:
        return _OrmConverter(topic, FromTaskConfig()())

    @factory.register(Instruction)
    def _default_instruction_converter(
        topic: str, data: Instruction, **kwargs: Any
    ) -> _OrmConverter[Instruction]:
        return _OrmConverter(topic, FromInstructionConfig()())

    @factory.register(Episode)
    def _default_episode_converter(
        topic: str, data: Episode, **kwargs: Any
    ) -> _OrmConverter[Episode]:
        return _OrmConverter(topic, FromEpisodeConfig()())

    return factory


class _OrmConverter(Generic[T]):
    def __init__(self, topic: str, converter: Any) -> None:
        self._topic = topic
        self._converter = converter

    def convert(self, message: StampedMessage[T]) -> MessageDict:
        data = self._converter.convert(message.data)
        return {
            self._topic: [
                StampedMessage(
                    data=data,
                    log_time=message.log_time,
                    pub_time=message.pub_time,
                )
            ]
        }
