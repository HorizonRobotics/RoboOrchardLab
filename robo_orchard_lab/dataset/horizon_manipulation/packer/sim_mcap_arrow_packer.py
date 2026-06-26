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

import argparse
import bisect
import glob
import hashlib
import json
import logging
import os
from typing import Generator, List, Literal

import datasets as hg_datasets
import numpy as np
import torch
from pydantic import Field
from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncoded,
    BatchCameraDataEncodedFeature,
    BatchFrameTransform,
    BatchJointsState,
    BatchJointsStateFeature,
    Distortion,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_split import (
    MakeIterMsgArgs,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    McapDecoderContext,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import McapReader
from robo_orchard_lab.dataset.horizon_manipulation.packer.sim_mcap_arrow_config import (  # noqa: E501
    load_episode_records_from_json,
)
from robo_orchard_lab.dataset.horizon_manipulation.packer.utils import (
    PackConfig,
    ParseConfig,
    filter_frames,
    get_failed_frames,
    get_static_frames,
    get_urdf_with_custom_fields,
    scale_images_and_update_intrinsics,
    time_sync,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    DatasetPackaging,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    RobotDescriptionFormat,
    TaskData,
)

# Setup Logger
logger = LoggerManager().get_child(__name__)
logger.setLevel(logging.INFO)


NORMALIZED_PARENT_FRAME_IDS = {
    "robots/dualarm_piper/left_base_link",
    "robots/dualarm_piperx/left_base_link",
    "robots/franka_panda/panda_link0",
}


def _normalize_parent_frame_id(
    parent_frame_id: str, camera_tf_topic: str
) -> str:
    if parent_frame_id not in NORMALIZED_PARENT_FRAME_IDS:
        raise ValueError(
            f"Unsupported parent_frame_id {parent_frame_id!r} "
            f"for camera tf topic {camera_tf_topic!r}. Expected one of "
            f"{sorted(NORMALIZED_PARENT_FRAME_IDS)!r}."
        )
    return "world"


# ========= Arrow Dataset Feature Definition =========
def make_dataset_features(camera_names: List[str]) -> hg_datasets.Features:
    features = {
        "uuid": hg_datasets.Value("string"),
        "joints": BatchJointsStateFeature(dtype="float32"),
        "actions": BatchJointsStateFeature(dtype="float32"),
    }
    for camera_name in camera_names:
        features[camera_name] = BatchCameraDataEncodedFeature(dtype="float32")
        features[f"{camera_name}_depth"] = BatchCameraDataEncodedFeature(
            dtype="float32"
        )
    return hg_datasets.Features(features)


class SimMcapParseConfig(ParseConfig):
    CAMERAS: List[str] = Field(
        ...,
        description="List of camera names",
    )

    # Topic configurations
    ROBOT: Literal["piper", "piperx", "franka"] = Field(
        ..., description="Robot preset used to parse sim MCAP data."
    )
    ARM_JOINT_SPECS: List["ArmJointSpec"] = Field(
        ...,
        min_length=1,
        description=(
            "Per-arm joint topic specs with slave_joint, master_joint, "
            "and optional master_gripper_joint_name."
        ),
    )
    JOINT_DIM: int = Field(
        ..., description="Number of joints encoded per arm output."
    )

    TF_STATIC: List[str] = Field(
        ..., description="Static transform topics for configured cameras."
    )

    COLOR_IMAGE_TOPICS: List[str] = Field(
        ...,
        description="List of color image topics for all cameras",
    )

    DEPTH_IMAGE_TOPICS: List[str] = Field(
        ...,
        description="List of depth image topics for all cameras",
    )

    COLOR_INFO_TOPICS: List[str] = Field(
        ...,
        description="List of color info topics for all cameras",
    )

    DEPTH_INFO_TOPICS: List[str] = Field(
        ...,
        description="List of depth info topics for all cameras",
    )
    META_TOPICS: List[str] = Field(
        default_factory=lambda: ["/observation/meta"],
        description="List of meta info topics",
    )
    META_KEYS: List[str] = Field(
        default_factory=lambda: ["init_position"],
        description="List of meta info keys to extract",
    )


class ArmJointSpec(ParseConfig):
    slave_joint: str | List[str] = Field(
        ..., description="Observation joint topic or topic group for one arm."
    )
    master_joint: str | List[str] = Field(
        ..., description="Action joint topic or topic group for one arm."
    )
    master_gripper_joint_name: str | None = Field(
        default=None,
        description=(
            "Optional gripper joint name used when merging arm "
            "and gripper topics."
        ),
    )


def make_piper_parse_config() -> SimMcapParseConfig:
    camera_names = ["static_camera", "left_hand_camera", "right_hand_camera"]
    camera_topics = [
        _camera_topics(camera_name) for camera_name in camera_names
    ]
    return SimMcapParseConfig(
        ROBOT="piper",
        CAMERAS=list(camera_names),
        ARM_JOINT_SPECS=[
            ArmJointSpec(
                slave_joint="/robot/left/joint_states",
                master_joint=[
                    "/action/robot_state/left_joint/joint_states",
                    "/action/robot_state/left_gripper/joint_states",
                ],
                master_gripper_joint_name="joint1",
            ),
            ArmJointSpec(
                slave_joint="/robot/right/joint_states",
                master_joint=[
                    "/action/robot_state/right_joint/joint_states",
                    "/action/robot_state/right_gripper/joint_states",
                ],
                master_gripper_joint_name="joint1",
            ),
        ],
        JOINT_DIM=7,
        COLOR_IMAGE_TOPICS=[item["color_image"] for item in camera_topics],
        DEPTH_IMAGE_TOPICS=[item["depth_image"] for item in camera_topics],
        COLOR_INFO_TOPICS=[item["color_info"] for item in camera_topics],
        DEPTH_INFO_TOPICS=[item["depth_info"] for item in camera_topics],
        TF_STATIC=[item["tf_static"] for item in camera_topics],
    )


def make_franka_parse_config() -> SimMcapParseConfig:
    source_cameras = ["ext1_camera", "ext2_camera", "wrist_camera"]
    camera_topics = [
        _camera_topics(camera_name) for camera_name in source_cameras
    ]
    return SimMcapParseConfig(
        ROBOT="franka",
        CAMERAS=list(source_cameras),
        ARM_JOINT_SPECS=[
            ArmJointSpec(
                slave_joint="/robot/joint_states",
                master_joint=[
                    "/action/robot_state/joint/joint_states",
                    "/action/robot_state/gripper/joint_states",
                ],
                master_gripper_joint_name="joint1",
            )
        ],
        JOINT_DIM=8,
        COLOR_IMAGE_TOPICS=[item["color_image"] for item in camera_topics],
        DEPTH_IMAGE_TOPICS=[item["depth_image"] for item in camera_topics],
        COLOR_INFO_TOPICS=[item["color_info"] for item in camera_topics],
        DEPTH_INFO_TOPICS=[item["depth_info"] for item in camera_topics],
        TF_STATIC=[item["tf_static"] for item in camera_topics],
        META_TOPICS=["/meta_data"],
    )


def make_piperx_parse_config() -> SimMcapParseConfig:
    return make_piper_parse_config().model_copy(update={"ROBOT": "piperx"})


def _format_joint_timestamp(timestamp) -> int:
    if isinstance(timestamp, int):
        return timestamp
    if hasattr(timestamp, "ToNanoseconds"):
        return int(timestamp.ToNanoseconds())
    if hasattr(timestamp, "seconds") and hasattr(timestamp, "nanos"):
        return int(timestamp.seconds * 1e9) + int(timestamp.nanos)
    raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")


def _format_time_list(ts_list: list) -> List[int]:
    return [_format_joint_timestamp(ts) for ts in ts_list]


def _normalize_joint_topics(topics: str | List[str] | None) -> List[str]:
    if topics is None:
        return []
    if isinstance(topics, str):
        return [topics]
    return topics


def _collect_joint_topics(parse_config: SimMcapParseConfig) -> List[str]:
    topics: List[str] = []
    for arm_spec in parse_config.ARM_JOINT_SPECS:
        topics.extend(_normalize_joint_topics(arm_spec.slave_joint))
        topics.extend(_normalize_joint_topics(arm_spec.master_joint))
    return topics


def _scale_gripper_joint(position: np.ndarray, joint_dim: int) -> np.ndarray:
    if position.shape[1] >= joint_dim:
        position[:, joint_dim - 1] = position[:, joint_dim - 1] * 2
    return position


def _make_batch_joint_state(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    effort: np.ndarray,
    names: List[str],
    timestamps: List[int],
    joint_dim: int,
) -> BatchJointsState:
    position = _scale_gripper_joint(position=position, joint_dim=joint_dim)
    return BatchJointsState(
        position=torch.from_numpy(position).to(dtype=torch.float32),
        velocity=torch.from_numpy(velocity).to(dtype=torch.float32),
        effort=torch.from_numpy(effort).to(dtype=torch.float32),
        names=names,
        timestamps=timestamps,
    )


def _camera_topics(camera_name: str) -> dict[str, str]:
    camera_prefix = f"/observation/cameras/{camera_name}"
    return {
        "color_image": f"{camera_prefix}/color_image/image_raw",
        "depth_image": f"{camera_prefix}/depth_image/image_raw",
        "color_info": f"{camera_prefix}/color_image/camera_info",
        "depth_info": f"{camera_prefix}/depth_image/camera_info",
        "tf_static": f"{camera_prefix}/color_image/tf",
    }


def build_arm_joint_batch(
    msgs: list,
    joint_dim: int,
) -> BatchJointsState:
    position = np.array(
        [[joint_state.position for joint_state in msg.states] for msg in msgs]
    )[:, :joint_dim]
    velocity = np.array(
        [[joint_state.velocity for joint_state in msg.states] for msg in msgs]
    )[:, :joint_dim]
    effort = np.array(
        [[joint_state.effort for joint_state in msg.states] for msg in msgs]
    )[:, :joint_dim]
    joint_ts_ns = _format_time_list([msg.timestamp for msg in msgs])
    joint_names = [
        joint_state.name for joint_state in msgs[0].states[:joint_dim]
    ]
    return _make_batch_joint_state(
        position=position,
        velocity=velocity,
        effort=effort,
        names=joint_names,
        timestamps=joint_ts_ns,
        joint_dim=joint_dim,
    )


def build_master_arm_gripper_joint_batch(
    *,
    arm_msgs: list,
    gripper_msgs: list,
    joint_dim: int,
    gripper_joint_name: str,
) -> BatchJointsState:
    if not arm_msgs or not gripper_msgs:
        raise ValueError(
            "no frames found for arm or gripper joint topics: "
            f"{len(arm_msgs)} arm, {len(gripper_msgs)} gripper"
        )
    if len(arm_msgs) != len(gripper_msgs):
        raise ValueError(
            "inconsistent frame counts across arm and gripper topics: "
            f"{len(arm_msgs)} != {len(gripper_msgs)}"
        )
    if joint_dim < 1:
        raise ValueError("joint_dim must be positive")

    arm_joint_dim = joint_dim - 1
    position = []
    velocity = []
    effort = []
    names = []

    for idx, (arm_msg, gripper_msg) in enumerate(
        zip(arm_msgs, gripper_msgs, strict=True)
    ):
        arm_states = arm_msg.states[:arm_joint_dim]
        if len(arm_states) != arm_joint_dim:
            raise ValueError(
                f"arm message at index {idx} has {len(arm_states)} joints, "
                f"expected {arm_joint_dim}"
            )

        gripper_state = next(
            (
                joint_state
                for joint_state in gripper_msg.states
                if joint_state.name == gripper_joint_name
            ),
            None,
        )
        if gripper_state is None:
            raise KeyError(
                f"Missing gripper joint '{gripper_joint_name}' at index {idx}"
            )

        position.append(
            [joint_state.position for joint_state in arm_states]
            + [gripper_state.position]
        )
        velocity.append(
            [joint_state.velocity for joint_state in arm_states]
            + [gripper_state.velocity]
        )
        effort.append(
            [joint_state.effort for joint_state in arm_states]
            + [gripper_state.effort]
        )

        if not names:
            names = [joint_state.name for joint_state in arm_states] + [
                gripper_state.name
            ]

    joint_ts_ns = _format_time_list([msg.timestamp for msg in arm_msgs])
    return _make_batch_joint_state(
        position=np.array(position),
        velocity=np.array(velocity),
        effort=np.array(effort),
        names=names,
        timestamps=joint_ts_ns,
        joint_dim=joint_dim,
    )


def _build_joint_batch(
    *,
    topics: str | List[str],
    joints: dict[str, list],
    joint_dim: int,
    gripper_joint_name: str | None = None,
) -> BatchJointsState:
    normalized_topics = _normalize_joint_topics(topics)
    if isinstance(topics, str):
        return build_arm_joint_batch(
            msgs=joints[topics],
            joint_dim=joint_dim,
        )

    if len(normalized_topics) == 2:
        if gripper_joint_name is not None:
            return build_master_arm_gripper_joint_batch(
                arm_msgs=joints[normalized_topics[0]],
                gripper_msgs=joints[normalized_topics[1]],
                joint_dim=joint_dim,
                gripper_joint_name=gripper_joint_name,
            )

    raise ValueError(
        "Unsupported joint topic layout. Expected either a single joint "
        "state topic or an arm+gripper topic pair with "
        "master_gripper_joint_name configured."
    )


def _build_joint_batches(
    *,
    parse_config: SimMcapParseConfig,
    joints: dict[str, list],
) -> tuple[list[BatchJointsState], list[BatchJointsState]]:
    observation_joint_batches: list[BatchJointsState] = []
    action_joint_batches: list[BatchJointsState] = []
    for arm_spec in parse_config.ARM_JOINT_SPECS:
        observation_joint_batches.append(
            _build_joint_batch(
                topics=arm_spec.slave_joint,
                joints=joints,
                joint_dim=parse_config.JOINT_DIM,
            )
        )
        action_joint_batches.append(
            _build_joint_batch(
                topics=arm_spec.master_joint,
                joints=joints,
                joint_dim=parse_config.JOINT_DIM,
                gripper_joint_name=arm_spec.master_gripper_joint_name,
            )
        )
    return observation_joint_batches, action_joint_batches


def _joint_batches_to_dict(
    prefix: str,
    joint_batches: list[BatchJointsState],
) -> dict[str, BatchJointsState]:
    return {
        f"{prefix}_{index}": joint_batch
        for index, joint_batch in enumerate(joint_batches)
    }


def _make_output_joint_features(
    *,
    parse_config: SimMcapParseConfig,
    observation_joint_batches: list[BatchJointsState],
    action_joint_batches: list[BatchJointsState],
) -> tuple[BatchJointsState, BatchJointsState]:
    arm_specs = parse_config.ARM_JOINT_SPECS
    if len(arm_specs) == 1:
        return observation_joint_batches[0], action_joint_batches[0]

    return (
        BatchJointsState.concat(observation_joint_batches, dim=1),
        BatchJointsState.concat(action_joint_batches, dim=1),
    )


class SubtaskIndex:
    def __init__(self, json_path, skip_first=True):
        # Load raw data
        with open(json_path, "r") as f:
            data = json.load(f)

        self.result = {}
        last_entry_by_obj = {}
        last_active_obj = None
        just_finished_stage = None

        # -------- Skip first action -------
        if skip_first:
            self.result["init"] = [
                {
                    "stage": "init",
                    "start_ts": data[0]["start_ts"],
                    "end_ts": data[0]["end_ts"],
                }
            ]
            data = data[1:]

        # -------- Build result dict --------
        for item in data:
            task = item["task"]
            obj = item["target_object"]
            start_ts = item["start_ts"]
            end_ts = item["end_ts"]

            if task in ["PrePick", "PICK", "PLACE"]:
                if obj not in self.result:
                    self.result[obj] = []

                entry = {
                    "stage": task,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                }
                self.result[obj].append(entry)

                last_entry_by_obj[obj] = entry
                last_active_obj = obj
                just_finished_stage = task

            elif task == "MOVE":
                merged = False

                if last_active_obj and just_finished_stage in [
                    "PICK",
                    "PLACE",
                ]:
                    last_entry_by_obj[last_active_obj]["end_ts"] = end_ts
                    merged = True

                if not merged and last_active_obj:
                    obj = last_active_obj
                    if obj not in self.result:
                        self.result[obj] = []
                    entry = {
                        "stage": "BACK",
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    }
                    self.result[obj].append(entry)
                    last_entry_by_obj[obj] = entry

                just_finished_stage = None

        # -------- Build interval index (for fast search) --------
        self.intervals = []
        for obj, recs in self.result.items():
            for r in recs:
                self.intervals.append(
                    (
                        r["start_ts"],
                        r["end_ts"],
                        obj,
                        r["stage"],
                    )
                )

        self.intervals.sort(key=lambda x: x[0])
        self.starts = [it[0] for it in self.intervals]
        self.obj_category_cache = self.get_obj_category()

    # -------- Query timestamp --------
    def query(self, ts):
        idx = bisect.bisect_right(self.starts, ts) - 1
        if idx < 0:
            return None

        start_ts, end_ts, obj, stage = self.intervals[idx]
        if start_ts <= ts <= end_ts:
            return {"object": obj, "stage": stage}

        return None

    def get_obj_category(self):
        obj_category = {}
        urdf_root = "/home/users/mengao.zhao/docker_env/ubuntu2204_orchard_sim_dev/asset3d-gen/tmp_outputs/"  # TODO: replace with a configured asset root.  # noqa: E501
        objs = os.listdir(urdf_root)
        for obj in objs:
            urdf_paths = glob.glob(os.path.join(urdf_root, obj, "*.urdf"))
            if not urdf_paths:
                raise FileNotFoundError(
                    f"No URDF found under {os.path.join(urdf_root, obj)}"
                )
            urdf_path = urdf_paths[0]
            _, custom_data = get_urdf_with_custom_fields(
                urdf_path, "extra_info"
            )
            category = custom_data["category"]
            obj_category[obj] = category
        return obj_category


def parse_sim_mcap(parse_config: SimMcapParseConfig, mcap_path: str):
    """Loads, synchronizes, and filters all data from the MCAP file.

    This is the core data processing method. It performs the following
    steps:
    1.  Finds and opens the MCAP file for the episode.
    2.  Reads all relevant ROS messages (images, joints, TF, etc.).
    """

    joint_topics = _collect_joint_topics(parse_config)

    all_topics = (
        parse_config.COLOR_IMAGE_TOPICS
        + parse_config.DEPTH_IMAGE_TOPICS
        + parse_config.COLOR_INFO_TOPICS
        + parse_config.DEPTH_INFO_TOPICS
        + parse_config.TF_STATIC
        + joint_topics
        + parse_config.META_TOPICS
    )

    tfs, images, depths, joints, image_infos, depth_infos, metas = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for topic in parse_config.COLOR_IMAGE_TOPICS:
        images[topic] = []
        image_infos[topic] = []
    for topic in parse_config.DEPTH_IMAGE_TOPICS:
        depths[topic] = []
        depth_infos[topic] = []
    for topic in joint_topics:
        joints[topic] = []
    for topic in parse_config.TF_STATIC:
        tfs[topic] = []
    for key in parse_config.META_KEYS:
        metas[key] = {}

    with open(mcap_path, "rb") as f:
        reader = McapReader.make_reader(f)
        for msg_tuple in reader.iter_decoded_messages(
            decoder_ctx=McapDecoderContext(),
            iter_config=MakeIterMsgArgs(topics=all_topics),
        ):
            msg, topic = msg_tuple.decoded_message, msg_tuple.channel.topic
            if topic in images:
                images[topic].append(msg)
            elif topic in depths:
                depths[topic].append(msg)
            elif topic in joints:
                joints[topic].append(msg)
            elif topic in parse_config.COLOR_INFO_TOPICS:
                idx = parse_config.COLOR_INFO_TOPICS.index(topic)
                img_topic = parse_config.COLOR_IMAGE_TOPICS[idx]
                image_infos[img_topic].append(msg)
            elif topic in parse_config.DEPTH_INFO_TOPICS:
                idx = parse_config.DEPTH_INFO_TOPICS.index(topic)
                dpt_topic = parse_config.DEPTH_IMAGE_TOPICS[idx]
                depth_infos[dpt_topic].append(msg)
            elif topic in parse_config.TF_STATIC:
                tfs[topic].append(msg)
            elif topic in parse_config.META_TOPICS:
                for meta_key in parse_config.META_KEYS:
                    if meta_key not in msg:
                        logger.info(f"{meta_key} not in meta info message.")
                        continue
                    for i in list(msg[meta_key].keys()):
                        metas[meta_key][i] = np.array(msg[meta_key][i])

    assert image_infos.keys() == images.keys()
    assert depth_infos.keys() == depths.keys()

    # Transfer Image to BatchCameraDataEncoded
    batch_image_dict = {}
    for topic in images:
        frame_num = len(images[topic])

        assert len(image_infos[topic]) == frame_num
        intrinsic = [image_infos[topic][idx].P for idx in range(frame_num)]
        intrinsic = np.array(intrinsic).reshape(frame_num, 3, 4)
        intrinsic_matrix = torch.from_numpy(intrinsic[..., :3, :3]).to(
            dtype=torch.float32
        )

        camera_tf_topic = topic.replace("/image_raw", "/tf")
        assert len(tfs[camera_tf_topic]) == frame_num
        parent_frame_id = _normalize_parent_frame_id(
            tfs[camera_tf_topic][0].parent_frame_id, camera_tf_topic
        )

        child_frame_id = tfs[camera_tf_topic][0].child_frame_id
        xyz = torch.tensor(
            [
                [item.translation.x, item.translation.y, item.translation.z]
                for item in tfs[camera_tf_topic]
            ]
        )
        quat = torch.tensor(
            [
                [
                    item.rotation.w,
                    item.rotation.x,
                    item.rotation.y,
                    item.rotation.z,
                ]
                for item in tfs[camera_tf_topic]
            ]
        )
        pose = BatchFrameTransform(
            parent_frame_id=parent_frame_id,
            child_frame_id=child_frame_id,
            xyz=xyz,
            quat=quat,
        )

        frame_id = images[topic][0].frame_id
        image_hw = (image_infos[topic][0].height, image_infos[topic][0].width)
        distortion = Distortion(
            model="plumb_bob", coefficients=torch.zeros(5, dtype=torch.float32)
        )
        sensor_data = [images[topic][idx].data for idx in range(frame_num)]
        image_ts_ns = [
            images[topic][idx].timestamp for idx in range(frame_num)
        ]
        image_ts_ns = _format_time_list(image_ts_ns)

        batch_image_msg = BatchCameraDataEncoded(
            topic=topic,
            frame_id=frame_id,
            image_shape=image_hw,
            intrinsic_matrices=intrinsic_matrix,
            pose=pose,
            distortion=distortion,
            sensor_data=sensor_data,
            format="jpeg",
            timestamps=image_ts_ns,
        )
        batch_image_dict[topic] = batch_image_msg

    batch_depth_dict = {}
    for topic in depths:
        frame_num = len(depths[topic])

        if len(depths[topic]) == len(depth_infos[topic]):
            intrinsic = [depth_infos[topic][idx].P for idx in range(frame_num)]
            intrinsic = np.array(intrinsic).reshape(frame_num, 3, 4)
            intrinsic_matrix = torch.from_numpy(intrinsic[..., :3, :3]).to(
                dtype=torch.float32
            )
        else:
            intrinsic = depth_infos[topic][0].p
            intrinsic = np.array(intrinsic).reshape(3, 4)
            intrinsic_matrix = (
                torch.from_numpy(intrinsic[..., :3, :3])
                .repeat(frame_num, 1, 1)
                .to(dtype=torch.float32)
            )

        image_hw = (depth_infos[topic][0].height, depth_infos[topic][0].width)
        frame_id = depths[topic][0].frame_id
        distortion = Distortion(
            model="plumb_bob", coefficients=torch.zeros(5, dtype=torch.float32)
        )
        sensor_data = [depths[topic][idx].data for idx in range(frame_num)]
        image_ts_ns = [
            depths[topic][idx].timestamp for idx in range(frame_num)
        ]
        image_ts_ns = _format_time_list(image_ts_ns)

        batch_depth_msg = BatchCameraDataEncoded(
            topic=topic,
            frame_id=frame_id,
            image_shape=image_hw,
            intrinsic_matrices=intrinsic_matrix,
            distortion=distortion,
            sensor_data=sensor_data,
            format="png",
            timestamps=image_ts_ns,
        )
        batch_depth_dict[topic] = batch_depth_msg

    # Transfer Joint to BatchJointsState
    observation_joint_batches, action_joint_batches = _build_joint_batches(
        parse_config=parse_config,
        joints=joints,
    )

    return (
        observation_joint_batches,
        action_joint_batches,
        batch_image_dict,
        batch_depth_dict,
        metas,
    )


# ========= Core Packaging Class =========
class McapEpisodePackaging(EpisodePackaging):
    """Processes single robotics episode from an MCAP file for Arrow packaging.

    This class handles the entire pipeline for a single episode: loading data
    from an MCAP file, synchronizing multimodal data streams (images, joints),
    filtering unwanted frames, and preparing the data for generation of a
    Hugging Face dataset in Arrow format.

    The synchronization strategy uses a primary camera's timestamps as the
    structural base, ensuring that each frame in the output dataset corresponds
    to a consistent point in time. Other data streams are aligned to this base
    clock by selecting the nearest temporal neighbors.
    """

    def __init__(
        self,
        episode_path: str,
        user: str,
        task_name: str,
        instruction: str,
        date: str,
        urdf_path: str,
        pack_config: PackConfig,
    ):
        """Initializes the McapEpisodePackaging instance.

        Args:
            episode_path: The path to the episode directory.
            user: The user associated with the episode.
            task_name: The name of the task performed.
            date: The recording date of the episode.
            urdf_path: A string containing the robot's URDF.
            pack_config: Configuration parameters for packaging.
        """
        self.episode_path = episode_path
        self.user = user
        self.task_name = task_name
        self.instruction = instruction
        self.date = date
        self.pack_config = pack_config
        self.urdf_path = urdf_path
        self.mcap_path = self.episode_path
        self.uuid = hashlib.md5(self.mcap_path.encode("utf-8")).hexdigest()

    def generate_episode_meta(self) -> EpisodeMeta:
        """Generates the metadata for the current episode.

        This method compiles the necessary metadata objects that describe the
        episode, the robot used, and the task performed, which will be stored
        alongside the Arrow dataset.

        Returns:
            EpisodeMeta: An object containing the structured metadata for the
                episode.
        """

        # Init Metadata
        urdf_content = open(self.urdf_path, "r").read()
        return EpisodeMeta(
            episode=EpisodeData(
                info={
                    "uuid": self.uuid,
                    "date": self.date,
                }
            ),
            robot=RobotData(
                name=os.path.basename(self.urdf_path),
                content=urdf_content,
                content_format=RobotDescriptionFormat.URDF,
            ),
            task=TaskData(
                name=self.task_name,
                description=self.instruction,
            ),
        )

    def process_data(self):
        logger.info(f"Start processing episode: {self.uuid}")
        pack_config: PackConfig = self.pack_config
        parse_config: SimMcapParseConfig = pack_config.PARSE_CONFIG

        # Parse MCAP
        mcap_data = parse_sim_mcap(
            mcap_path=self.episode_path, parse_config=parse_config
        )
        (
            observation_joint_batches,
            action_joint_batches,
            batch_image_dict,
            batch_depth_dict,
            metas,
        ) = mcap_data

        # Time Synchronization
        base_time = batch_image_dict[pack_config.SYNC_CAMERA].timestamps
        time_sync(
            data=[
                _joint_batches_to_dict(
                    "observation_joint", observation_joint_batches
                ),
                _joint_batches_to_dict("action_joint", action_joint_batches),
                batch_image_dict,
                batch_depth_dict,
            ],
            base_time=base_time,
        )
        _, action_states = _make_output_joint_features(
            parse_config=parse_config,
            observation_joint_batches=observation_joint_batches,
            action_joint_batches=action_joint_batches,
        )

        # Add subtask descriptions
        if pack_config.WITH_SUBTASK:
            with_arm_info = True
            subtask_info = SubtaskIndex(
                json_path=self.episode_path.replace(
                    "env0_data.mcap", "task_status.json"
                )
            )
            subtasks = []
            for i in base_time:
                q = subtask_info.query(i)
                if q is None:
                    subtask_ = ""
                else:
                    stage = q["stage"]
                    if stage == "init":
                        subtask_ = "go to init position"
                    # elif stage in ["PrePick", "PICK", "PLACE"]:
                    #     obj = q["object"].split("/")[-1].split("_")[-1]
                    #     obj_category = subtask_info.obj_category_cache.get(
                    #         obj, "object"
                    #     )
                    #     if stage == "PICK":
                    #         subtask_ = f"grasp {obj_category.lower()}."
                    #     elif stage == "PLACE":
                    #         subtask_ = (
                    #             f"place {obj_category.lower()} into basket."
                    #         )
                    # elif stage == "BACK":
                    #     subtask_ = "move back to default position."
                    else:
                        # only support pick-place task for now
                        obj = q["object"].split("/")[-1].split("_")[-1]
                        obj_category = subtask_info.obj_category_cache.get(
                            obj, "object"
                        )
                        if stage not in ["PrePick", "PICK", "PLACE", "BACK"]:
                            raise ValueError(f"Unknown subtask stage: {stage}")
                        subtask_ = f"Grasp the {obj_category.lower()} and place it into the basket."  # noqa: E501
                        if with_arm_info:
                            y = metas["init_position"][f"obj_{obj}"][1]
                            if y > 0:
                                subtask_ = f"Using left arm, grasp the {obj_category.lower()} and place it into the basket."  # noqa: E501
                            else:
                                subtask_ = f"Using right arm, grasp the {obj_category.lower()} and place it into the basket."  # noqa: E501
                subtasks.append(subtask_)

        # Filter static frames
        retained_index = get_static_frames(
            joint_positions=action_states.position,
            base_time=base_time,
            static_threshold=pack_config.STATIC_THRESHOLD,
            head_time_to_filter=pack_config.HEAD_TIME_TO_FILTER,
            tail_time_to_filter=pack_config.TAIL_TIME_TO_FILTER,
        )

        # Filter failed frames
        if pack_config.FILTER_FAILED_FRAME:
            task_status_path = self.episode_path.replace(
                "env0_data.mcap", "task_status.json"
            )
            if not os.path.exists(task_status_path):
                logger.error(f"task_status file not found: {task_status_path}")
            task_status = json.load(open(task_status_path, "r"))

            status_retained_index = get_failed_frames(
                task_status,
                base_time=base_time,
                skip_first=True,
            )
            retained_index = retained_index & status_retained_index

        logger.info(
            f"Filtering: {len(base_time)} steps -> {retained_index.sum()} steps"  # noqa: E501
        )
        filter_frames(
            data=[
                _joint_batches_to_dict(
                    "observation_joint", observation_joint_batches
                ),
                _joint_batches_to_dict("action_joint", action_joint_batches),
                batch_image_dict,
                batch_depth_dict,
            ],
            retained_index=retained_index,
        )
        if pack_config.WITH_SUBTASK:
            subtasks = [
                i for idx, i in enumerate(subtasks) if retained_index[idx]
            ]
            self.subtasks = subtasks
        else:
            self.subtasks = [""] * retained_index.sum()

        # Scale images and update intrinsics
        scale_images_and_update_intrinsics(
            image_data=[batch_image_dict, batch_depth_dict],
            image_scale_ratio=pack_config.IMAGE_SCALE,
        )
        joint_states, action_states = _make_output_joint_features(
            parse_config=parse_config,
            observation_joint_batches=observation_joint_batches,
            action_joint_batches=action_joint_batches,
        )

        self.observation_joint_batches = observation_joint_batches
        self.action_joint_batches = action_joint_batches
        self.batch_image_dict = batch_image_dict
        self.batch_depth_dict = batch_depth_dict
        self.joint_states = joint_states
        self.action_states = action_states

        self.mcap_path = self.episode_path
        self.base_time = batch_image_dict[pack_config.SYNC_CAMERA].timestamps
        self.num_steps = len(self.base_time)

    def generate_frames(self) -> Generator[DataFrame, None, None]:
        """Generates structured DataFrame for each frame in the episode.

        This generator function iterates through each processed time step of
        the episode and yields a `DataFrame` object. Each `DataFrame` contains
        all the data for that specific moment, including joint states (robot),
        actions (teleop controller), multiple camera views, and instructional
        data.

        Yields:
            DataFrame: An object representing all data for a single,
                synchronized frame of the episode.
        """

        def get_index_camera(data: BatchCameraDataEncoded, index):
            ret_dict = {}
            for key in [
                "topic",
                "frame_id",
                "image_shape",
                "distortion",
                "format",
            ]:
                ret_dict[key] = getattr(data, key)
            ret_dict["sensor_data"] = [data.sensor_data[index]]
            ret_dict["timestamps"] = [data.timestamps[index]]
            ret_dict["intrinsic_matrices"] = data.intrinsic_matrices[
                index : index + 1
            ]

            if data.pose is not None:
                pose = BatchFrameTransform(
                    parent_frame_id=data.pose.parent_frame_id,
                    child_frame_id=data.pose.child_frame_id,
                    xyz=data.pose.xyz[index : index + 1],
                    quat=data.pose.quat[index : index + 1],
                )
                ret_dict["pose"] = pose

            ret = BatchCameraDataEncoded(**ret_dict)
            return ret

        self.process_data()
        if self.num_steps == 0:
            logger.warning(
                f"Episode {self.uuid} has 0 steps after processing."
            )
            return

        instruction = InstructionData(
            name=self.task_name,
            json_content={
                "name": self.task_name,
                "description": self.instruction,
            },
        )

        parse_config: SimMcapParseConfig = self.pack_config.PARSE_CONFIG
        # --- Camera Data ---
        for i in range(self.num_steps):
            features = {
                "uuid": self.uuid,
                "joints": self.joint_states[i],
                "actions": self.action_states[i],
            }
            for cam_idx, cam_name in enumerate(parse_config.CAMERAS):
                color_topic = parse_config.COLOR_IMAGE_TOPICS[cam_idx]
                depth_topic = parse_config.DEPTH_IMAGE_TOPICS[cam_idx]
                features[cam_name] = get_index_camera(
                    self.batch_image_dict[color_topic], i
                )
                features[f"{cam_name}_depth"] = get_index_camera(
                    self.batch_depth_dict[depth_topic], i
                )

            frame_ts_ns = self.base_time[i]

            instruction.json_content["subtask"] = self.subtasks[i]

            yield DataFrame(
                features=features,
                instruction=instruction,
                timestamp_ns_max=frame_ts_ns,
                timestamp_ns_min=frame_ts_ns,
            )


def make_dataset_from_mcap(
    input_path: str,
    output_path: str,
    urdf_path: str,
    task_name: str,
    instruction: str,
    pack_config: PackConfig,
    max_shard_size: str | int = "2GB",
    split: hg_datasets.Split | None = None,
    force_overwrite: bool = False,
):
    """Orchestrates conversion of multiple MCAP episodes into an Arrow dataset.

    This function reads a JSON file of episode records, initializes a
    `McapEpisodePackaging` instance for each valid episode, and then uses the
    `DatasetPackaging` utility to write all episodes into a sharded Arrow
    dataset compatible with Hugging Face datasets.

    Args:
        input_path (str): Path to a JSON array file. Each item must contain
            `mcap_path` and `instruction`, for example:
            `[{"mcap_path": "/path/to/env0_data.mcap", "instruction":
            "Pick the red marker."}]`.
        output_path (str): The destination path for the output Arrow dataset.
        urdf_path (str): The path to the robot's URDF file.
        pack_config (PackConfig): Configuration parameters for packaging,
        max_shard_size (str | int): The maximum size for each Arrow file shard.
            Defaults to "2GB".
        split (hg_datasets.Split | None): The dataset split to assign (e.g.,
            'train', 'test'). Defaults to None, which the packager typically
            treats as 'train'.
        force_overwrite (bool): If True, the destination directory will be
            overwritten if it already exists. Defaults to False.
    """
    episodes_meta = []
    logger.info(f"Loading episode records from JSON: {input_path}")
    if instruction:
        logger.warning(
            "--instruction is ignored for JSON-driven sim MCAP packing."
        )

    episode_records = load_episode_records_from_json(input_path)
    episode_records.sort(key=lambda item: item["mcap_path"])

    for record in episode_records:
        episode_path = record["mcap_path"]
        episode_instruction = record["instruction"]
        date = os.path.basename(os.path.dirname(os.path.dirname(episode_path)))
        user = "default_user"
        task = task_name
        episodes_meta.append(
            [episode_path, user, task, date, episode_instruction]
        )

    episodes_meta.sort()
    logger.info(f"Found {len(episodes_meta)} potential episodes.")

    episodes = []
    for (
        episode_path,
        user,
        task_name,
        date,
        episode_instruction,
    ) in episodes_meta:
        logger.info(
            "Processing episode: %s with instruction: %s",
            episode_path,
            episode_instruction,
        )
        try:
            packer = McapEpisodePackaging(
                episode_path=episode_path,
                user=user,
                task_name=task_name,
                instruction=episode_instruction,
                date=date,
                urdf_path=urdf_path,
                pack_config=pack_config,
            )
            episodes.append(packer)
        except Exception:
            logger.error(
                f"Failed to process episode at {episode_path}", exc_info=True
            )

    if not episodes:
        logger.error("No valid episodes found to package. Aborting.")
        return
    logger.info(f"Packaging {len(episodes)} valid episodes into Arrow format.")

    packing = DatasetPackaging(
        features=make_dataset_features(pack_config.PARSE_CONFIG.CAMERAS)
    )
    packing.packaging(
        episodes=episodes,
        dataset_path=output_path,
        max_shard_size=max_shard_size,
        force_overwrite=force_overwrite,
        split=split,
    )
    logger.info(f"Successfully created Arrow dataset at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MCAP files directly to Arrow format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=(
            "Path to a JSON array file. Each item must be an object with "
            "'mcap_path' and 'instruction', for example: "
            '[{"mcap_path": "/path/to/env0_data.mcap", '
            '"instruction": "Pick the red marker."}]'
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output Arrow dataset.",
    )
    parser.add_argument(
        "--urdf_path", type=str, required=True, help="Path to the URDF file."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task for the dataset.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Deprecated for this script. JSON item instructions are used.",
    )
    parser.add_argument(
        "--image_scale_factor",
        type=float,
        default=1.0,
        help="Factor to scale images by.",
    )
    parser.add_argument(
        "--robot",
        choices=["piper", "piperx", "franka"],
        required=True,
        help="Robot preset to use when parsing sim MCAP files.",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Overwrite the output directory if it exists.",
    )
    args = parser.parse_args()

    if args.robot == "franka":
        parse_config = make_franka_parse_config()
    elif args.robot == "piperx":
        parse_config = make_piperx_parse_config()
    else:
        parse_config = make_piper_parse_config()
    if not parse_config.COLOR_IMAGE_TOPICS:
        raise ValueError(
            f"COLOR_IMAGE_TOPICS must be configured for robot {args.robot}"
        )
    pack_config = PackConfig(
        SYNC_CAMERA=parse_config.COLOR_IMAGE_TOPICS[0],
        IMAGE_SCALE=args.image_scale_factor,
        STATIC_THRESHOLD=1e-3,
        HEAD_TIME_TO_FILTER=None,
        TAIL_TIME_TO_FILTER=None,
        PARSE_CONFIG=parse_config,
        FILTER_FAILED_FRAME=False,
        WITH_SUBTASK=False,
    )

    make_dataset_from_mcap(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf_path=args.urdf_path,
        task_name=args.task_name,
        instruction=args.instruction,
        pack_config=pack_config,
        force_overwrite=args.force_overwrite,
    )
