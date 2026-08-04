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
from collections.abc import Iterator

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
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterStateful,
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


class _ConversionError(RuntimeError):
    pass


class _SourceError(RuntimeError):
    pass


class _CleanupError(RuntimeError):
    pass


class _RecordingStatefulConverter(MessageConverterStateful[int, str]):
    def __init__(
        self,
        *,
        fail_on: int | None = None,
        cleanup_error: BaseException | None = None,
    ):
        self.calls: list[tuple[str, int | None]] = []
        self._fail_on = fail_on
        self._cleanup_error = cleanup_error

    def convert(self, src: int | None) -> Iterator[str]:
        self.calls.append(("convert", src))
        if self._fail_on is not None and src == self._fail_on:
            raise _ConversionError(f"cannot convert {src}")
        yield f"converted:{src}"

    def flush(self) -> list[str]:
        self.calls.append(("flush", None))
        if self._cleanup_error is not None:
            raise self._cleanup_error
        return ["flushed"]


def test_make_iterator_preserves_normal_finalization_order() -> None:
    """Normal exhaustion converts None before the explicit flush."""

    converter = _RecordingStatefulConverter()

    assert list(converter.make_iterator(iter([1, 2]))) == [
        "converted:1",
        "converted:2",
        "converted:None",
        "flushed",
    ]
    assert converter.calls == [
        ("convert", 1),
        ("convert", 2),
        ("convert", None),
        ("flush", None),
    ]


def test_make_iterator_flushes_after_source_failure() -> None:
    """A source exception stays primary while converter state is finalized."""

    converter = _RecordingStatefulConverter()

    def source() -> Iterator[int]:
        yield 1
        raise _SourceError("source failed")

    iterator = converter.make_iterator(source())
    assert next(iterator) == "converted:1"
    with pytest.raises(_SourceError, match="source failed"):
        next(iterator)
    assert converter.calls[-1] == ("flush", None)


def test_make_iterator_flushes_after_conversion_failure() -> None:
    """A conversion exception triggers eager converter finalization."""

    converter = _RecordingStatefulConverter(fail_on=2)

    with pytest.raises(_ConversionError, match="cannot convert 2"):
        list(converter.make_iterator(iter([1, 2])))
    assert converter.calls[-1] == ("flush", None)


def test_make_iterator_flushes_when_consumer_closes() -> None:
    """Closing a partial iterator eagerly finalizes the converter."""

    converter = _RecordingStatefulConverter()
    iterator = converter.make_iterator(iter([1, 2]))

    assert next(iterator) == "converted:1"
    iterator.close()

    assert converter.calls[-1] == ("flush", None)


def test_make_iterator_preserves_primary_cleanup_failure() -> None:
    """Cleanup failures are reported without replacing conversion failures."""

    converter = _RecordingStatefulConverter(
        fail_on=2,
        cleanup_error=_CleanupError("cleanup failed"),
    )

    with pytest.warns(UserWarning, match="cleanup failed"):
        with pytest.raises(_ConversionError, match="cannot convert 2"):
            list(converter.make_iterator(iter([1, 2])))


def test_make_iterator_append_none_without_flush() -> None:
    """append_none remains independent when the caller owns cleanup."""

    source_consumed = False

    def source() -> Iterator[int]:
        nonlocal source_consumed
        source_consumed = True
        yield 1

    converter = _RecordingStatefulConverter()
    iterator = converter.make_iterator(
        source(),
        append_none=True,
        flush=False,
    )

    assert list(iterator) == ["converted:1", "converted:None"]
    assert source_consumed
    assert ("flush", None) not in converter.calls


def test_make_iterator_flush_false_leaves_lifecycle_to_caller() -> None:
    """flush=False never finalizes the converter implicitly."""

    converter = _RecordingStatefulConverter()

    assert list(
        converter.make_iterator(
            iter([1, 2]),
            append_none=False,
            flush=False,
        )
    ) == ["converted:1", "converted:2"]
    assert ("flush", None) not in converter.calls


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
