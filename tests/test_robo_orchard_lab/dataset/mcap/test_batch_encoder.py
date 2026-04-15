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

import importlib
import os
import warnings

import fsspec
import pytest
import torch

from robo_orchard_lab.dataset.experimental.mcap.batch_decoder import (
    McapBatch2BatchCameraDataEncodedConfig,
    McapBatch2BatchJointStateConfig,
    McapBatch2BatchPoseConfig,
    McapBatchDecoderConfig,
    McapBatchDecoders,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchEncoderConfig,
    McapBatchEncoders,
    McapBatchFromBatchCameraDataEncodedConfig,
    McapBatchFromBatchJointStateConfig,
    McapBatchFromBatchPoseConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    McapMessageBatch,
    SplitBatchByTopicArgs,
    SplitBatchByTopics,
    iter_messages_batch,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_encoder.encoder_ctx import (  # noqa: E501
    FoxgloveEncoder,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    McapReader,
)


def tensor_equal(
    src: torch.Tensor | None, dst: torch.Tensor | None, atol: float = 1e-8
) -> bool:
    """Check if two tensors are equal within a tolerance.

    Args:
        src (torch.Tensor | None): The first tensor.
        dst (torch.Tensor | None): The second tensor.
        eps (float, optional): The tolerance for equality. Defaults to 1e-6.

    Returns:
        bool: True if the tensors are equal within the tolerance,
            False otherwise.
    """
    if [src, dst].count(None) == 1:
        return False
    if src is None and dst is None:
        return True
    assert src is not None and dst is not None
    return torch.allclose(src, dst, atol)


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


@pytest.fixture(scope="module")
def example_msg_batch(example_reader: McapReader):
    # Create a batch of messages from the example reader
    for batch in iter_messages_batch(
        example_reader,
        batch_split=SplitBatchByTopics(
            [
                SplitBatchByTopicArgs(
                    monitor_topic="/observation/cameras/wrist/left/image",
                    min_messages_per_topic=3,
                    max_messages_per_topic=3,
                )
            ]
        ),
    ):
        yield batch
        break


class TestBatchFromDataDict:
    @pytest.mark.parametrize(
        "decoder_configs, encoder_configs",
        [
            (
                {
                    "joints": McapBatch2BatchJointStateConfig(
                        source_topic="/action/robot_state/joints"
                    ),
                    "camera_encoded": McapBatch2BatchCameraDataEncodedConfig(
                        image_topic="/observation/cameras/wrist/left/image",
                        tf_topic="/observation/cameras/wrist/left/tf",
                        calib_topic="/observation/cameras/wrist/left/camera_calib",
                    ),
                    "ee_pose": McapBatch2BatchPoseConfig(
                        source_topic="/action/robot_state/ee_pose"
                    ),
                },
                {
                    "joints": McapBatchFromBatchJointStateConfig(
                        target_topic="/action/robot_state/joints"
                    ),
                    "camera_encoded": McapBatchFromBatchCameraDataEncodedConfig(  # noqa: E501
                        image_topic="/observation/cameras/wrist/left/image",
                        tf_topic="/observation/cameras/wrist/left/tf",
                        calib_topic="/observation/cameras/wrist/left/camera_calib",
                    ),
                    "ee_pose": McapBatchFromBatchPoseConfig(
                        target_topic="/action/robot_state/ee_pose"
                    ),
                },
            )
        ],
    )
    def test_batch_from_data_dict(
        self,
        example_msg_batch: McapMessageBatch,
        decoder_configs: dict[str, McapBatchDecoderConfig],
        encoder_configs: dict[str, McapBatchEncoderConfig],
    ):
        # Create a decoder factory with the converter
        to_data = McapBatchDecoders(decoder_configs)
        example_msg_batch.sort()
        with McapDecoderContext() as msg_decoder_ctx:
            decoded_msg = to_data(
                example_msg_batch, msg_decoder_ctx=msg_decoder_ctx
            )
        for k in decoder_configs.keys():
            assert k in decoded_msg, f"Key {k} not found in the result"
            print(f"{k}: {type(decoded_msg[k])}")
        from_data = McapBatchEncoders(encoder_configs)
        formatted_batch = from_data.format_batch(decoded_msg)

        foxglove_encoded_batch = from_data(
            decoded_msg, msg_encoder_ctx=FoxgloveEncoder(None)
        )
        assert set(foxglove_encoded_batch.keys()) == set(
            formatted_batch.keys()
        )
        for topic, encoded_msgs in foxglove_encoded_batch.items():
            assert topic in formatted_batch
            assert len(encoded_msgs.messages) == len(formatted_batch[topic])

    @pytest.mark.parametrize(
        "decoder_configs, encoder_configs",
        [
            (
                {
                    "joints": McapBatch2BatchJointStateConfig(
                        source_topic="/action/robot_state/joints"
                    ),
                    "camera_encoded": McapBatch2BatchCameraDataEncodedConfig(
                        image_topic="/observation/cameras/wrist/left/image",
                        tf_topic="/observation/cameras/wrist/left/tf",
                        calib_topic="/observation/cameras/wrist/left/camera_calib",
                    ),
                    "ee_pose": McapBatch2BatchPoseConfig(
                        source_topic="/action/robot_state/ee_pose"
                    ),
                },
                {
                    "joints": McapBatchFromBatchJointStateConfig(
                        target_topic="/action/robot_state/joints"
                    ),
                    "camera_encoded": McapBatchFromBatchCameraDataEncodedConfig(  # noqa: E501
                        image_topic="/observation/cameras/wrist/left/image",
                        tf_topic="/observation/cameras/wrist/left/tf",
                        calib_topic="/observation/cameras/wrist/left/camera_calib",
                    ),
                    "ee_pose": McapBatchFromBatchPoseConfig(
                        target_topic="/action/robot_state/ee_pose"
                    ),
                },
            )
        ],
    )
    def test_deprecated_protobuf_encoder_round_trip_compatibility(
        self,
        example_msg_batch: McapMessageBatch,
        decoder_configs: dict[str, McapBatchDecoderConfig],
        encoder_configs: dict[str, McapBatchEncoderConfig],
    ):
        """Keep protobuf compatibility coverage."""
        to_data = McapBatchDecoders(decoder_configs)
        example_msg_batch.sort()
        with McapDecoderContext() as msg_decoder_ctx:
            decoded_msg = to_data(
                example_msg_batch, msg_decoder_ctx=msg_decoder_ctx
            )
        from_data = McapBatchEncoders(encoder_configs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            msg_encoder_ctx = importlib.import_module(
                "robo_orchard_lab.dataset.experimental.mcap.msg_encoder"
            ).McapProtobufEncoder()
        reconstructed_batch = McapMessageBatch(
            message_dict=from_data(
                decoded_msg, msg_encoder_ctx=msg_encoder_ctx
            )
        )
        reconstructed_batch.sort()

        # We does not check the reconstructed mcap messages
        # because there may be some differences due to numeric
        # precision
        # Instead, we check the decoded data from the reconstructed batch
        # and the original decoded data
        with McapDecoderContext() as msg_decoder_ctx:
            new_decoded_msg = to_data(
                reconstructed_batch, msg_decoder_ctx=msg_decoder_ctx
            )

        for k in decoded_msg.keys():
            assert k in new_decoded_msg, (
                f"Key {k} not found in the original batch"
            )
            decoded = decoded_msg[k]
            new_decoded = new_decoded_msg[k]

            assert decoded == new_decoded, (
                f"Decoded data for key {k} does not match after encoding and "
                f"decoding. Original: {decoded}, New: {new_decoded}"
            )
