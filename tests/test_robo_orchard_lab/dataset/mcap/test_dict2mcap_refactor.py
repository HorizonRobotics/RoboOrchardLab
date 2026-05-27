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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from robo_orchard_lab.dataset.datatypes import BatchCameraData, ImageMode
from robo_orchard_lab.dataset.experimental.mcap.foxglove_writer import (
    FoxgloveMcapWriter as McapWriter,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
    McapReader,
)
from robo_orchard_lab.dataset.experimental.mcap.topics import (
    camera_image_topic,
)
from robo_orchard_lab.dataset.experimental.mcap.writer import (
    Dict2Mcap,
    StampedMessage,
    ToMcapMessageFactory,
)


def _read_topics_and_log_times(path: Path) -> list[tuple[str, int]]:
    with path.open("rb") as f:
        reader = McapReader.make_reader(f)
        return [
            (msg.topic, msg.message.log_time)
            for msg in reader.iter_messages(
                MakeIterMsgArgs(log_time_order=False)
            )
        ]


def _camera_data() -> BatchCameraData:
    return BatchCameraData(
        sensor_data=torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
        frame_id="front",
        pix_fmt=ImageMode.RGB,
        timestamps=[100],
    )


def test_save_to_mcap_writes_input_order_without_log_time_merge(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input_order.mcap"
    data = {
        "/later": [StampedMessage(data={"value": "later"}, log_time=20)],
        "/earlier": [StampedMessage(data={"value": "earlier"}, log_time=10)],
    }

    Dict2Mcap().save_to_mcap(data, mcap=path)

    assert _read_topics_and_log_times(path) == [
        ("/later", 20),
        ("/earlier", 10),
    ]


def test_save_to_mcap_accepts_deprecated_mcap_path_keyword(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deprecated_keyword.mcap"
    data = {"/topic": [StampedMessage(data={"value": 1}, log_time=1)]}

    with pytest.warns(DeprecationWarning, match="mcap_path"):
        Dict2Mcap().save_to_mcap(data=data, mcap_path=path)

    assert _read_topics_and_log_times(path) == [("/topic", 1)]


def test_save_to_mcap_reuses_opened_writer(tmp_path: Path) -> None:
    path = tmp_path / "opened_writer.mcap"
    writer = Dict2Mcap()

    with path.open("wb") as f, McapWriter(f) as mcap_writer:
        writer.save_to_mcap(
            {"/first": [StampedMessage(data={"value": 1}, log_time=1)]},
            mcap=mcap_writer,
        )
        writer.save_to_mcap(
            {"/second": [StampedMessage(data={"value": 2}, log_time=2)]},
            mcap=mcap_writer,
        )

    assert _read_topics_and_log_times(path) == [
        ("/first", 1),
        ("/second", 2),
    ]


@dataclass
class _SourceBatch:
    values: list[int]
    local_log_times: list[int]


class _SourceBatchConverter:
    def __init__(self) -> None:
        self.converted: list[_SourceBatch] = []

    def convert(
        self, message: StampedMessage[_SourceBatch]
    ) -> dict[str, list[StampedMessage[Any]]]:
        self.converted.append(message.data)
        return {
            "/converted": [
                StampedMessage(data={"value": value}, log_time=log_time)
                for value, log_time in zip(
                    message.data.values,
                    message.data.local_log_times,
                    strict=True,
                )
            ]
        }


def test_converter_is_created_once_per_source_topic_and_anchor_aligned(
    tmp_path: Path,
) -> None:
    path = tmp_path / "converter.mcap"
    created: list[_SourceBatchConverter] = []
    factory = ToMcapMessageFactory()

    @factory.register(_SourceBatch)
    def create_converter(
        topic: str, data: _SourceBatch, **kwargs: Any
    ) -> _SourceBatchConverter:
        assert topic == "/source"
        assert data.values == [1, 2]
        assert kwargs == {}
        converter = _SourceBatchConverter()
        created.append(converter)
        return converter

    Dict2Mcap(converter_factory=factory).save_to_mcap(
        {
            "/source": [
                StampedMessage(
                    data=_SourceBatch(
                        values=[1, 2], local_log_times=[100, 110]
                    ),
                    log_time=1000,
                ),
                StampedMessage(
                    data=_SourceBatch(values=[3, 4], local_log_times=[50, 60]),
                    log_time=2000,
                ),
            ]
        },
        mcap=path,
    )

    assert len(created) == 1
    assert [batch.values for batch in created[0].converted] == [
        [1, 2],
        [3, 4],
    ]
    assert _read_topics_and_log_times(path) == [
        ("/converted", 1000),
        ("/converted", 1010),
        ("/converted", 2000),
        ("/converted", 2010),
    ]


def test_default_camera_converter_uses_neutral_image_topic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "camera_neutral_topic.mcap"
    source_topic = "sample/images/front"

    Dict2Mcap().save_to_mcap(
        {source_topic: [StampedMessage(data=_camera_data(), log_time=100)]},
        mcap=path,
    )

    assert _read_topics_and_log_times(path) == [
        (camera_image_topic(source_topic), 100)
    ]


def test_writer_converters_apply_after_dict2mcap_expansion(
    tmp_path: Path,
) -> None:
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    path = tmp_path / "camera_compressed.mcap"
    source_topic = "sample/images/front"
    image_topic = camera_image_topic(source_topic)

    Dict2Mcap(
        writer_converters={
            image_topic: RawImage2CompressedImageConfig(format="png")
        }
    ).save_to_mcap(
        {source_topic: [StampedMessage(data=_camera_data(), log_time=100)]},
        mcap=path,
    )

    with path.open("rb") as f:
        messages = list(
            McapReader.make_reader(f).iter_messages(
                MakeIterMsgArgs(topics=[image_topic])
            )
        )

    assert len(messages) == 1
    assert messages[0].schema is not None
    assert messages[0].schema.name == CompressedImage.DESCRIPTOR.full_name


def test_writer_converters_reject_already_open_writer(
    tmp_path: Path,
) -> None:
    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    path = tmp_path / "opened_writer_policy.mcap"
    source_topic = "sample/images/front"
    image_topic = camera_image_topic(source_topic)
    writer = Dict2Mcap(
        writer_converters={
            image_topic: RawImage2CompressedImageConfig(format="png")
        }
    )

    with path.open("wb") as f, McapWriter(f) as mcap_writer:
        with pytest.raises(TypeError, match="McapWriter constructor"):
            writer.save_to_mcap(
                {
                    source_topic: [
                        StampedMessage(data=_camera_data(), log_time=100)
                    ]
                },
                mcap=mcap_writer,
            )


def test_open_writer_converters_apply_after_dict2mcap_expansion(
    tmp_path: Path,
) -> None:
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage

    from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
        RawImage2CompressedImageConfig,
    )

    path = tmp_path / "opened_writer_compressed.mcap"
    source_topic = "sample/images/front"
    image_topic = camera_image_topic(source_topic)

    with (
        path.open("wb") as f,
        McapWriter(
            f,
            converters={
                image_topic: RawImage2CompressedImageConfig(format="png")
            },
        ) as mcap_writer,
    ):
        Dict2Mcap().save_to_mcap(
            {
                source_topic: [
                    StampedMessage(data=_camera_data(), log_time=100)
                ]
            },
            mcap=mcap_writer,
        )

    with path.open("rb") as f:
        messages = list(
            McapReader.make_reader(f).iter_messages(
                MakeIterMsgArgs(topics=[image_topic])
            )
        )

    assert len(messages) == 1
    assert messages[0].schema is not None
    assert messages[0].schema.name == CompressedImage.DESCRIPTOR.full_name
