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

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
from robo_orchard_core.datatypes import (
    BatchCameraData,
    BatchFrameTransform,
    BatchJointsState,
    ImageMode,
)
from robo_orchard_core.utils.math import (
    Transform3D_M,
    check_valid_rotation_matrix,
)

from robo_orchard_lab.dataset.datatypes import BatchFrameTransformGraph
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.robot.db_orm import Robot

if TYPE_CHECKING:
    from robo_orchard_lab.envs.robotwin.kinematics import RoboTwinEEF

__all__ = [
    "get_camera_data",
    "get_joints",
    "get_observation_cams",
]

_LEFT_EEF_FROM_JOINT_FRAME_ID = "left_eef_from_joint"
_RIGHT_EEF_FROM_JOINT_FRAME_ID = "right_eef_from_joint"


def _build_mcap_observation_messages(
    *,
    obs: dict[str, Any],
    topic_prefix: str,
    log_time: int,
    instruction_payload: Any | None,
    meta_payload: Any,
) -> dict[str, list[StampedMessage[Any]]]:
    """Convert an already-cached observation and payloads to MCAP messages."""
    messages: dict[str, list[StampedMessage[Any]]] = {}

    cameras = obs.get("cameras") or {}
    for camera_name, camera_streams in cameras.items():
        for stream_name, camera_data in camera_streams.items():
            if not isinstance(camera_data, BatchCameraData):
                continue
            topic = f"{topic_prefix}/cameras/{camera_name}/{stream_name}"
            timestamps = [log_time] * camera_data.batch_size
            messages[topic] = [
                StampedMessage(
                    data=camera_data.model_copy(
                        update={"timestamps": timestamps}
                    ),
                    log_time=log_time,
                    pub_time=log_time,
                )
            ]

    joints = obs.get("joints")
    if isinstance(joints, BatchJointsState):
        timestamps = [log_time] * joints.batch_size
        messages[f"{topic_prefix}/joints"] = [
            StampedMessage(
                data=joints.model_copy(update={"timestamps": timestamps}),
                log_time=log_time,
                pub_time=log_time,
            )
        ]

    tf_graph = obs.get("tf")
    if isinstance(tf_graph, BatchFrameTransformGraph):
        tf_state = tf_graph.as_state()
        if tf_state.tf_list:
            messages[f"{topic_prefix}/tf"] = [
                StampedMessage(
                    data=BatchFrameTransformGraph(
                        tf_list=[
                            tf.model_copy(
                                update={
                                    "timestamps": [log_time] * tf.batch_size
                                }
                            )
                            for tf in tf_state.tf_list
                        ],
                        bidirectional=tf_state.bidirectional,
                        static_tf=tf_state.static_tf,
                    ),
                    log_time=log_time,
                    pub_time=log_time,
                )
            ]

    if instruction_payload is not None:
        messages[f"{topic_prefix}/instruction"] = [
            StampedMessage(
                data=instruction_payload,
                log_time=log_time,
                pub_time=log_time,
            )
        ]

    messages[f"{topic_prefix}/meta"] = [
        StampedMessage(
            data=meta_payload,
            log_time=log_time,
            pub_time=log_time,
        )
    ]

    robots = obs.get("robots")
    if isinstance(robots, dict):
        for robot_name, robot in robots.items():
            if isinstance(robot_name, str) and isinstance(robot, Robot):
                messages[f"{topic_prefix}/meta/robots/{robot_name}"] = [
                    StampedMessage(
                        data=robot,
                        log_time=log_time,
                        pub_time=log_time,
                    )
                ]

    return messages


def _format_observation(
    ret: dict[str, Any],
    *,
    instructions: object,
    step_index: int,
    step_timestamp: float,
    format_datatypes: bool,
    joint_names: list[str] | None,
    base_tf_graph: BatchFrameTransformGraph,
    robots: dict[str, Robot],
    joint_eef: RoboTwinEEF | None,
    left_control_eef_frame_id: str,
    right_control_eef_frame_id: str,
) -> dict[str, Any]:
    """Format one owned raw RoboTwin observation in place.

    Base and EEF transforms are world-frame edges. Split-runtime control EEF
    frame IDs are already namespaced by the caller-provided runtime layout.
    Joint-derived EEF edges use stable adapter-owned frame IDs so they cannot
    collide with RoboTwin's raw control-pose frames.
    """
    eef_tf_edges: list[BatchFrameTransform] = []
    if joint_eef is not None:
        eef_tf_edges.extend(
            [
                joint_eef.left_eef.model_copy(
                    update={"child_frame_id": _LEFT_EEF_FROM_JOINT_FRAME_ID}
                ),
                joint_eef.right_eef.model_copy(
                    update={"child_frame_id": _RIGHT_EEF_FROM_JOINT_FRAME_ID}
                ),
            ]
        )

    endpose = ret.get("endpose")
    if isinstance(endpose, dict) and endpose:
        eef_tf_edges.extend(
            [
                _pose_vector_to_tf(
                    endpose["left_endpose"],
                    child_frame_id=left_control_eef_frame_id,
                ),
                _pose_vector_to_tf(
                    endpose["right_endpose"],
                    child_frame_id=right_control_eef_frame_id,
                ),
            ]
        )

    ret["instructions"] = instructions
    ret["step_index"] = step_index
    ret["step_timestamp"] = step_timestamp
    if format_datatypes:
        ret["joints"] = get_joints(ret, joint_names=joint_names)
        ret.pop("joint_action", None)
        ret["cameras"] = get_observation_cams(ret)
        ret.pop("observation")
    ret["tf"] = base_tf_graph
    ret["robots"] = robots
    if eef_tf_edges:
        ret["tf"].add_tf(eef_tf_edges)
    return ret


def _extract_video_frame(raw_obs: dict[str, Any]) -> np.ndarray | None:
    """Return a contiguous RGB head-camera frame when one is available."""
    observation = raw_obs.get("observation")
    if not isinstance(observation, dict):
        return None
    head_camera = observation.get("head_camera")
    if not isinstance(head_camera, dict):
        return None
    frame = head_camera.get("rgb")
    if frame is None:
        return None

    frame_np = np.asarray(frame)
    if frame_np.ndim != 3 or frame_np.shape[2] != 3:
        return None
    if frame_np.dtype != np.uint8:
        frame_np = frame_np.astype(np.uint8)
    return np.ascontiguousarray(frame_np)


def _pose_vector_to_tf(
    pose_vector: Sequence[float] | np.ndarray,
    *,
    child_frame_id: str,
) -> BatchFrameTransform:
    """Convert a RoboTwin EE pose vector to a world-frame transform.

    Args:
        pose_vector (Sequence[float] | np.ndarray): RoboTwin EE pose in
            ``[x, y, z, qw, qx, qy, qz]`` order.
        child_frame_id (str): Child frame name for the returned transform.

    Returns:
        BatchFrameTransform: ``world -> child_frame_id`` transform.
    """
    pose_np = np.asarray(pose_vector, dtype=np.float32)
    if pose_np.shape != (7,):
        raise ValueError(
            "Expected RoboTwin endpose to contain 7 values "
            f"(xyz + quaternion), got shape {tuple(pose_np.shape)}."
        )
    return BatchFrameTransform(
        xyz=torch.from_numpy(pose_np[:3]).unsqueeze(0),
        quat=torch.from_numpy(pose_np[3:]).unsqueeze(0),
        parent_frame_id="world",
        child_frame_id=child_frame_id,
    )


def get_joints(
    obs: dict,
    joint_names: list[str] | None = None,
    timestamps: list[int] | None = None,
) -> BatchJointsState:
    """Convert joint data from observation dict to BatchJointsState.

    This function assumes the input observation dict has the
    following structure (RoboTwin):

    .. code-block:: text

        {
            "joint_action": {
                "vector": (N,) np.ndarray,
                ...
            },
            ... other keys ...
        }

    """
    arr = torch.from_numpy(obs["joint_action"]["vector"]).unsqueeze(0)
    return BatchJointsState(
        position=arr, names=joint_names, timestamps=timestamps
    )


def get_observation_cams(obs: dict) -> dict:
    """Convert all camera data in the observation dict.

    This function assumes the input observation dict has
    the following structure (RoboTwin):

    .. code-block:: text

        {
            "observation": {
                "camera_name_1": { ... camera data ... },
                "camera_name_2": { ... camera data ... },
                ...
            },
            ... other keys ...
        }

    Each camera data dict follows the same format as described
    in `get_camera_data`.

    Example output structure::

        {
            "camera_name_1": {
                "rgb": BatchCameraData,
                "depth": BatchCameraData,
            },
            "camera_name_2": {
                "rgb": BatchCameraData,
            },
            ...
        }

    Returns:
        dict: Mapping from camera name to converted BatchCameraData entries.

    """
    ret = {}
    for k, v in obs["observation"].items():
        ret[k] = get_camera_data(v, camera_name=k)
    return ret


def get_camera_data(
    cam: dict, camera_name: str, timestamps: list[int] | None = None
) -> dict[str, BatchCameraData]:
    """Convert camera data from dict to BatchCameraData.

        This function assumes the input camera dict has the following keys:

        - ``rgb``: ``(H, W, 3)`` np.ndarray, optional.
        - ``depth``: ``(H, W)`` np.ndarray, optional.
        - ``intrinsic_cv``: ``(3, 3)`` np.ndarray.
        - ``extrinsic_cv``: ``(4, 4)`` np.ndarray camera extrinsic matrix.
            Uses the OpenCV convention. It encodes the camera pose in the
            world frame after inversion.

    Args:
        cam (dict): Camera data in dict format.
        camera_name (str): Name of the camera.

    Returns:
        dict[str, BatchCameraData]: Converted camera data keyed by stream type.

    """
    # `extrinsic_cv` follows the external OpenCV camera extrinsic
    # convention. Invert it here to build the camera pose in world.
    world_to_camera_mat = torch.eye(4).reshape(1, 4, 4)
    world_to_camera_mat[0, :3, :] = torch.from_numpy(cam["extrinsic_cv"])
    camera_in_world = Transform3D_M(matrix=world_to_camera_mat).inverse()
    assert check_valid_rotation_matrix(
        camera_in_world.get_matrix()[:, :3, :3], tol=1e-5
    ), "Invalid camera_in_world rotation matrix"
    cam_tf = BatchFrameTransform(
        xyz=camera_in_world.get_translation(),
        quat=camera_in_world.get_rotation_quaternion(),
        parent_frame_id="world",
        child_frame_id=camera_name,
    )
    intrinsic = torch.from_numpy(cam["intrinsic_cv"]).unsqueeze(0)

    ret = {}

    if "rgb" in cam:
        ret["rgb"] = BatchCameraData(
            sensor_data=torch.from_numpy(cam["rgb"]).unsqueeze(0),
            pix_fmt=ImageMode.RGB,
            frame_id=camera_name,
            intrinsic_matrices=intrinsic,
            pose=cam_tf,
            timestamps=timestamps,
        )

    if "depth" in cam:
        ret["depth"] = BatchCameraData(
            sensor_data=torch.from_numpy(cam["depth"])
            .unsqueeze(0)
            .to(dtype=torch.float32),
            pix_fmt=ImageMode.F,
            frame_id=camera_name,
            intrinsic_matrices=intrinsic,
            pose=cam_tf,
            timestamps=timestamps,
        )
    return ret
