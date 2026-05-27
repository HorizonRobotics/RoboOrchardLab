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

import cv2
import fsspec
import numpy as np
import pytest
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from google.protobuf.timestamp import from_nanoseconds
from robo_orchard_core.utils.torch_utils import make_device

from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    SplitBatchByTopicArgs,
    SplitBatchByTopics,
    iter_messages_batch,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    FgCameraCompressedImages,
    ToBatchCameraDataConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.joint_state import (  # noqa: E501
    ToBatchJointsStateConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    DecoderFactoryWithConverter,
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
    McapReader,
)


def _decode_compressed_image(data: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    return decoded


def test_raw_image_to_compressed_image_preserves_rgb_channel_order() -> None:
    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    raw = RawImage(
        timestamp=from_nanoseconds(1),
        frame_id="front",
        width=2,
        height=1,
        encoding="rgb8",
        step=8,
        data=bytes([255, 0, 0, 0, 255, 0, 99, 99]),
    )

    compressed = RawImage2CompressedImageConfig(format="png")().convert(raw)
    decoded = _decode_compressed_image(compressed.data)

    assert compressed.format == "png"
    assert compressed.frame_id == "front"
    assert decoded.tolist() == [[[0, 0, 255], [0, 255, 0]]]


def test_raw_image_to_compressed_image_preserves_bgr_channel_order() -> None:
    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    raw = RawImage(
        timestamp=from_nanoseconds(1),
        frame_id="front",
        width=2,
        height=1,
        encoding="bgr8",
        step=8,
        data=bytes([0, 0, 255, 0, 255, 0, 99, 99]),
    )

    compressed = RawImage2CompressedImageConfig(format="png")().convert(raw)
    decoded = _decode_compressed_image(compressed.data)

    assert decoded.tolist() == [[[0, 0, 255], [0, 255, 0]]]


def test_raw_image_to_compressed_image_rejects_float_raw_images() -> None:
    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    raw = RawImage(
        timestamp=from_nanoseconds(1),
        frame_id="front",
        width=1,
        height=1,
        encoding="32FC1",
        step=4,
        data=np.array([1.0], dtype=np.float32).tobytes(),
    )

    with pytest.raises(ValueError, match="32FC1"):
        RawImage2CompressedImageConfig(format="png")().convert(raw)


@pytest.fixture(scope="module")
def example_reader(ROBO_ORCHARD_TEST_WORKSPACE: str):
    mcap_file = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/mcap/RAIL+c3d50939+2023-11-20-17h-48m-24s_image.mcap",
    )
    with fsspec.open(mcap_file, "rb") as f:
        reader = McapReader.make_reader(f)  # type: ignore
        for _, channel in reader.get_summary().channels.items():  # type: ignore
            print(
                f"Channel: {channel.topic}, message_encoding: {channel.message_encoding}"  # noqa: E501
            )
        # for chunk_index in reader.get_summary().chunk_indexes:
        #     print(chunk_index)
        yield reader


class TestCameraMsgs2BatchCameraData:
    def test_camera_msgs_2_batch_camera_data(self, example_reader: McapReader):
        # Create a decoder factory with the converter
        factory = DecoderFactoryWithConverter()

        # # Create a context for the decoder
        context = McapDecoderContext([factory])
        as_compressed_image = ToBatchCameraDataConfig()()
        # Iterate over messages in batches
        img_topic = "/observation/cameras/wrist/left/image"
        for batch in iter_messages_batch(
            example_reader,
            batch_split=SplitBatchByTopics(
                SplitBatchByTopicArgs(
                    monitor_topic=img_topic,
                    min_messages_per_topic=3,
                    max_messages_per_topic=3,
                )
            ),
            iter_config=MakeIterMsgArgs(topics=[img_topic]),
        ):
            cur_batch_data = FgCameraCompressedImages(
                images=batch[img_topic].decode(context)
            )
            break

        assert len(cur_batch_data.images) == 3

        # Convert to BatchCameraDataWithTimestamps
        batch_camera_data = as_compressed_image.convert(cur_batch_data)
        assert batch_camera_data.timestamps is not None
        assert len(batch_camera_data.timestamps) == 3
        assert batch_camera_data.batch_size == 3
        assert (
            batch_camera_data.sensor_data.shape[1:3]
            == batch_camera_data.image_shape
        )
        assert batch_camera_data.pose is None
        assert batch_camera_data.distortion_model is None
        assert batch_camera_data.distorsion_coefficients is None

    def test_camera_msgs_2_batch_camera_data_with_tf_calib(
        self, example_reader: McapReader
    ):
        # Create a decoder factory with the converter
        factory = DecoderFactoryWithConverter()

        # # Create a context for the decoder
        context = McapDecoderContext([factory])
        as_compressed_image = ToBatchCameraDataConfig()()
        # Iterate over messages in batches
        img_topic = "/observation/cameras/wrist/left/image"
        # wrist camera tf(from baselink to camera_wrist_left)
        tf_topic = "/observation/cameras/wrist/left/tf"
        camera_calib_topic = "/observation/cameras/wrist/left/camera_calib"
        for batch in iter_messages_batch(
            example_reader,
            batch_split=SplitBatchByTopics(
                [
                    SplitBatchByTopicArgs(
                        monitor_topic=img_topic,
                        min_messages_per_topic=3,
                        max_messages_per_topic=3,
                    ),
                    SplitBatchByTopicArgs(
                        monitor_topic=tf_topic,
                        min_messages_per_topic=3,
                        max_messages_per_topic=3,
                    ),
                    SplitBatchByTopicArgs(
                        monitor_topic=camera_calib_topic,
                        min_messages_per_topic=1,
                        max_messages_per_topic=3,
                    ),
                ]
            ),
            iter_config=MakeIterMsgArgs(
                topics=[img_topic, tf_topic, camera_calib_topic]
            ),
        ):
            print(f"Batch size: {len(batch)}")
            cur_batch_data = FgCameraCompressedImages(
                images=batch[img_topic].decode(context),
                calib=batch[camera_calib_topic].decode(context)[0],
                tf=batch[tf_topic].decode(context),
            )
            break
        # Convert to BatchCameraDataWithTimestamps
        batch_camera_data = as_compressed_image.convert(cur_batch_data)
        assert batch_camera_data.timestamps is not None
        assert len(batch_camera_data.timestamps) == 3
        assert batch_camera_data.batch_size == 3
        assert (
            batch_camera_data.sensor_data.shape[1:3]
            == batch_camera_data.image_shape
        )
        assert batch_camera_data.pose is not None
        assert (
            batch_camera_data.pose.batch_size == batch_camera_data.batch_size
        )

        print(f"Pose: {batch_camera_data.pose}, ")
        assert batch_camera_data.distortion is not None
        print(f"Distortion: {batch_camera_data.distortion}, ")


class TestToBatchJointsState:
    @pytest.mark.parametrize(
        "device, topic",
        [
            ("cpu", "/action/robot_state/joints"),
            ("cuda:0", "/action/robot_state/joints"),
            ("cpu", "/action/robot_state/gripper"),
            ("cpu", "/action/robot_state/joint_torques_computed"),
        ],
    )
    def test_to_batch_joints_state(
        self, example_reader: McapReader, device: str, topic: str
    ):
        # Create a decoder factory with the converter
        factory = DecoderFactoryWithConverter()

        # Create a context for the decoder
        context = McapDecoderContext([factory])
        as_joint_state = ToBatchJointsStateConfig(device=device)()

        # Iterate over messages in batches
        joint_state_topic = topic
        for batch in iter_messages_batch(
            example_reader,
            batch_split=SplitBatchByTopics(
                SplitBatchByTopicArgs(
                    monitor_topic=joint_state_topic,
                    min_messages_per_topic=3,
                    max_messages_per_topic=3,
                )
            ),
            iter_config=MakeIterMsgArgs(topics=[joint_state_topic]),
        ):
            cur_batch_data = batch[joint_state_topic].decode(context)
            break

        # Convert to BatchJointsStateStamped
        batch_joints_state = as_joint_state.convert(cur_batch_data)
        assert batch_joints_state.batch_size == 3

        if batch_joints_state.position is not None:
            assert batch_joints_state.position.device == make_device(device)
        if batch_joints_state.velocity is not None:
            assert batch_joints_state.velocity.device == make_device(device)
        if batch_joints_state.effort is not None:
            assert batch_joints_state.effort.device == make_device(device)
        print(batch_joints_state)
