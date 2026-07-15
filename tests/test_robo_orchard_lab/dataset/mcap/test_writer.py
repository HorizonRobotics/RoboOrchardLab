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

import json
import os
import random
import string

import pytest

from robo_orchard_lab.dataset.datatypes import BatchCameraDataEncoded
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
    McapReader,
)
from robo_orchard_lab.dataset.experimental.mcap.writer import (
    Dataset2Mcap,
    Dict2Mcap,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import Episode


@pytest.fixture(scope="module")
def robotwin_dataset(ROBO_ORCHARD_TEST_WORKSPACE: str):
    dataset_dir = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/datasets/robotwin/ro_dataset",
    )
    dataset = RODataset(dataset_path=dataset_dir)
    yield dataset


class TestDataset2Mcap:
    def test_save_episode(
        self,
        robotwin_dataset: RODataset,
        tmp_local_folder: str,
    ):
        # Test saving an episode to MCAP format

        target_path = os.path.join(
            tmp_local_folder,
            "".join(random.choices(string.ascii_lowercase, k=8))
            + "_test.mcap",
        )
        to_mcap = Dataset2Mcap(dataset=robotwin_dataset)
        to_mcap.save_episode(
            target_path=target_path,
            episode_index=0,
            encoder_cfg={},
        )
        assert os.path.exists(target_path), "MCAP file was not created."

        episode = robotwin_dataset.get_meta(Episode, 0)
        assert episode is not None
        first_frame = robotwin_dataset.frame_dataset[
            episode.dataset_begin_index
        ]
        with open(target_path, "rb") as f:
            reader = McapReader.make_reader(f)
            episode_messages = list(
                reader.iter_messages(
                    MakeIterMsgArgs(topics=["/metadata/episode"])
                )
            )

        assert len(episode_messages) == 1
        assert episode_messages[0].channel.message_encoding == "json"
        episode_message = episode_messages[0].message
        assert episode_message.log_time == first_frame["timestamp_min"]
        episode_payload = json.loads(episode_message.data.decode("utf-8"))
        assert episode_payload == {
            "index": episode.index,
            "robot_index": episode.robot_index,
            "task_index": episode.task_index,
            "prev_episode_index": episode.prev_episode_index,
            "dataset_begin_index": episode.dataset_begin_index,
            "frame_num": episode.frame_num,
            "truncated": episode.truncated,
            "success": episode.success,
            "info": episode.info,
        }


class TestDict2Mcap:
    def test_dict2mcap(
        self,
        robotwin_dataset: RODataset,
        tmp_local_folder: str,
    ):
        # Test converting a dictionary to MCAP format

        # generate data dict
        data_dict = {}

        first_row = robotwin_dataset[0]
        meta = robotwin_dataset.convert_meta_index2meta(
            {
                "robot_index": first_row["robot_index"],
                "task_index": first_row["task_index"],
                "instruction_index": first_row["instruction_index"],
            }
        )
        print("meta: ", meta.keys())
        for k in ["robot", "task", "instruction"]:
            data_dict[k] = [StampedMessage(data=meta[k], log_time=0)]

        data_dict["joints"] = []
        data_dict["front_camera"] = []
        data_dict["front_camera_depth"] = []

        for row_id in range(10):
            row = robotwin_dataset[row_id]
            for k in row.keys():
                if k in data_dict:
                    data_dict[k].append(
                        StampedMessage(
                            data=row[k], log_time=int(row_id * 1e9 * 0.03)
                        )
                    )

        # add decoded camera data
        for camera_name in ["front_camera", "front_camera_depth"]:
            decoded_key = f"{camera_name}_decoded"
            src_msgs: list[StampedMessage[BatchCameraDataEncoded]] = data_dict[
                camera_name
            ]
            decoded_msgs = []
            for msg in src_msgs:
                decoded_data = msg.data.decode()
                decoded_msgs.append(
                    StampedMessage(
                        data=decoded_data,
                        log_time=msg.log_time,
                    )
                )
            data_dict[decoded_key] = decoded_msgs

        target_path = os.path.join(
            tmp_local_folder,
            "".join(random.choices(string.ascii_lowercase, k=8))
            + "_test.mcap",
        )

        dict2mcap = Dict2Mcap()
        dict2mcap.save_to_mcap(
            data=data_dict,
            mcap=target_path,
        )
