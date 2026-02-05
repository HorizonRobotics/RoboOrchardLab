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

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import datasets as hg_datasets
import numpy as np
import pytorch_kinematics as pk
import scipy
import torch
from arrow_base import BaseRODataPacker, ROPackConfig
from datasets.features import Array2D, Sequence, Value
from utils import (
    generate_static_mask,
    load_h5_to_dict,
    load_images_threaded,
    load_json,
    load_json_threaded,
    load_txt,
    save_json,
)
from pytorch3d.transforms import matrix_to_quaternion
from tqdm import tqdm

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraDataEncoded,
    BatchCameraDataEncodedFeature,
    BatchFrameTransform,
    BatchFrameTransformGraph,
    BatchFrameTransformGraphFeature,
    BatchJointsState,
    BatchJointsStateFeature,
)
from robo_orchard_lab.dataset.robot.packaging import (
    DataFrame,
    EpisodeData,
    EpisodeMeta,
    EpisodePackaging,
    InstructionData,
    RobotData,
    TaskData,
)

logger = logging.getLogger(__name__)


# from /GenieSimAssets/G1_omnipicker/G1_omnipicker.usda
# NOTE: default camera extrinsic in usda
robot_usda_configs = {
    "middle": {
        "usda_name": "Head_Camera",
        "parent_link": "head_link2",
        "xformOp:orient": (
            6.123233995736766e-17,
            0.7071067811865476,
            -4.329780281177467e-17,
            -0.7071067811865476,
        ),
        "xformOp:scale": (1, 1, 1),
        "xformOp:translate": (0.0858, -0.04119, 0),
        "xformOpOrder": [
            "xformOp:translate",
            "xformOp:orient",
            "xformOp:scale",
        ],
    },
    "left": {
        "usda_name": "Left_Camera",
        "parent_link": "gripper_l_base_link",
        "xformOp:orient": (
            -0.24763401411015198,
            -0.6623272567663911,
            0.6623272567663911,
            0.24763401411015207,
        ),
        "xformOp:scale": (1, 1, 1),
        "xformOp:translate": (-0.08248, -0.00244, 0.04346),
        "xformOpOrder": [
            "xformOp:translate",
            "xformOp:orient",
            "xformOp:scale",
        ],
    },
    "right": {
        "usda_name": "Right_Camera",
        "parent_link": "gripper_r_base_link",
        "xformOp:orient": (
            -0.24763401411015198,
            -0.6623272567663911,
            -0.6623272567663911,
            -0.24763401411015207,
        ),
        "xformOp:scale": (1, 1, 1),
        "xformOp:translate": (0.08248, -0.0024400039901093795, 0.04346),
        "xformOpOrder": [
            "xformOp:translate",
            "xformOp:orient",
            "xformOp:scale",
        ],
    },
}

# NOTE: default camera intrinsic
default_intrinsic = {
    "middle": {
        "fx": 634.0862399675711,
        "fy": 634.0862399675711,
        "ppx": 640.0,
        "ppy": 360.0,
        "distortion_model": "pinhole",
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
    "left": {
        "fx": 435.26140525224156,
        "fy": 435.26140525224156,
        "ppx": 424.0,
        "ppy": 240.0,
        "distortion_model": "pinhole",
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
    "right": {
        "fx": 435.26140525224156,
        "fy": 435.26140525224156,
        "ppx": 424.0,
        "ppy": 240.0,
        "distortion_model": "pinhole",
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
}


def get_static_tf_from_usda(usda_cfg):
    xyz = torch.tensor(usda_cfg["xformOp:translate"], dtype=torch.float32)
    quat_wxyz = torch.tensor(usda_cfg["xformOp:orient"], dtype=torch.float32)
    return xyz, quat_wxyz


@dataclass(frozen=True)
class AgibotDigitChallengePackConfig(ROPackConfig):
    """
    Docstring for AgibotDigitChallengePackConfig
    task_name_en: English name of the task
    available tasks:
    - "clear_table_in_the_restaurant"
    - "clear_the_countertop_waste"
    - "heat_the_food_in_the_microwave"
    - "make_a_sandwich"
    - "open_drawer_and_store_items"
    - "pack_in_the_supermarket"
    - "pack_moving_objects_from_conveyor"
    - "pickup_items_from_the_freezer"
    - "restock_supermarket_items"
    - "stamp_the_seal"
    - "" means all tasks
    """

    task_name: str = ""
    split: str = (
        "all"  # all, train, val - use 5% val for validation, hardcode use [::20]
    )
    skip_static_frames: bool = True
    static_threshold: float = 5e-4
    joint_dimensions: int = 20  # arm (8*2) + head (2) + body (2)
    storage_feature_type: str = "lite"  # "raw" or "compact" or "lite"
    lite_mode_hist_step: int = (
        8  # only for "lite" mode, including current step, must >=1
    )
    lite_mode_pred_step: int = 192  # only for "lite" mode
    image_max_w: int = 848  # max image for all pack mode
    image_max_h: int = 480


class AgibotDigitChallengeDataHelper:

    cam_names_src = ["head", "hand_left", "hand_right"]
    cam_names_dst = ["middle", "left", "right"]
    camera_num = len(cam_names_src)

    joint_dof = 7
    gripper_dof = 1
    joint_types = ["state", "action"]

    def __init__(
        self,
        cfg: AgibotDigitChallengePackConfig,
        meta: dict,
        urdf_content: str,
    ):

        meta_info_path = Path(meta["meta_info_path"])
        episode_dir = meta_info_path.parent

        proprio_states_path = episode_dir / "aligned_joints.h5"
        parameter_path = episode_dir / "parameters" / "camera"

        parameter = {}
        for cam_name_src in self.cam_names_src:
            intrinsic_path = (
                parameter_path / f"{cam_name_src}_intrinsic_params.json"
            )
            extrinsic_path = (
                parameter_path
                / f"{cam_name_src}_extrinsic_params_aligned.json"
            )
            parameter[cam_name_src] = {}
            if intrinsic_path.exists():
                parameter[cam_name_src]["intrinsic"] = load_json(
                    str(intrinsic_path)
                )
            else:
                camera_name_dst = self.cam_names_dst[
                    self.cam_names_src.index(cam_name_src)
                ]
                parameter[cam_name_src]["intrinsic"] = default_intrinsic[
                    camera_name_dst
                ]
            if extrinsic_path.exists():
                parameter[cam_name_src]["extrinsic"] = load_json(
                    str(extrinsic_path)
                )

        self.episode_dir = episode_dir
        self.proprio_states = load_h5_to_dict(str(proprio_states_path))
        self.parameter = parameter
        self.camera_dir = episode_dir / "camera"
        self.state_path = parameter_path / "state.json"

        # assert self.state_path.exists(), "state.json must exists!"
        if self.state_path.exists() is False:
            logger.warning(
                f"{str(self.state_path)} does not exist, use assets usda settings."
            )

        self.cfg = cfg
        self.meta = meta
        self.urdf_content = urdf_content

        timestamps_ns = self.proprio_states['timestamp'].astype(np.int64)
        self.timestamps_ns = timestamps_ns

    def _get_intrinsic_matrix(self, intrinsic: dict) -> list:
        fx, fy = intrinsic["fx"], intrinsic["fy"]
        ppx, ppy = intrinsic["ppx"], intrinsic["ppy"]
        intrinsic_matrix = [
            [fx, 0.0, ppx],
            [0.0, fy, ppy],
            [0.0, 0.0, 1.0],
        ]
        return intrinsic_matrix

    def _get_extrinsic_components(
        self, pose: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # xyzrpy: N x 6

        xyz = pose[:, :3, 3]
        quat_xyzw = scipy.spatial.transform.Rotation.from_matrix(
            pose[:, :3, :3]
        ).as_quat()  # x, y, z, w
        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]  # to w, x, y, z

        xyz = torch.tensor(xyz, dtype=torch.float32)
        quat_wxyz = torch.tensor(quat_wxyz, dtype=torch.float32)

        return xyz, quat_wxyz

    def _get_joint_names(self) -> list[str]:
        joint_names = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "gripper_hand_l",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "gripper_hand_r",
            "idx11_head_joint1",  # yaw
            "idx12_head_joint2",  # pitch
            "idx01_body_joint1",  # lift
            "idx02_body_joint2",  # pitch
        ]
        return joint_names

    def _map_to_fk_input(
        self, urdf_joint_names: list[str], batch_joint_state: BatchJointsState
    ) -> np.ndarray:
        assert (
            batch_joint_state.position is not None
        ), "Joint position data is None"
        assert batch_joint_state.names is not None, "Joint names data is None"

        joint_states = batch_joint_state.position.numpy()
        joint_names = batch_joint_state.names

        fk_input = np.zeros((joint_states.shape[0], len(urdf_joint_names)))
        for i, name in enumerate(urdf_joint_names):
            if name in joint_names:
                idx = joint_names.index(name)
                fk_input[:, i] = joint_states[:, idx]
            else:
                logger.warning(
                    f"Joint {name} not found in predefined joint names."
                )
        return fk_input

    def build_batch_color(self) -> dict[str, BatchCameraDataEncoded]:
        cam2color = {}
        img_idxes = os.listdir(self.camera_dir)
        img_idxes = sorted(img_idxes, key=lambda x: int(x))
        for i in range(self.camera_num):
            cam_name_src = self.cam_names_src[i]
            cam_name_dst = self.cam_names_dst[i]

            intrinsic = self.parameter[cam_name_src]["intrinsic"]
            intrinsic_matrix = self._get_intrinsic_matrix(intrinsic)
            intrinsic_matrix = torch.tensor(
                intrinsic_matrix, dtype=torch.float32
            )

            color_paths = [
                str(self.camera_dir / img_idx / f"{cam_name_src}_color.jpg")
                for img_idx in img_idxes
            ]

            color_imgs = load_images_threaded(color_paths)
            frame_id = f"{cam_name_dst}_camera_color_optical_frame"
            intrinsic_matrices = intrinsic_matrix.repeat(len(color_imgs), 1, 1)

            assert (
                color_imgs is not None
            ), f"Failed to load color images for {cam_name_src}"

            image_shape_hw = cv2.imread(
                color_paths[0], cv2.IMREAD_COLOR
            ).shape[:2]

            assert len(color_imgs) == len(self.timestamps_ns), (
                f"Frame number ({len(color_imgs)}) and timestamp number "
                f"({len(self.timestamps_ns)}) do not match in {str(self.camera_dir)}"
            )

            color_data = BatchCameraDataEncoded(
                topic=f"/observation/cameras/{cam_name_dst}/color_image/image_raw",
                frame_id=frame_id,
                image_shape=image_shape_hw,
                intrinsic_matrices=intrinsic_matrices,
                sensor_data=color_imgs,
                format="jpeg",
                timestamps=self.timestamps_ns.tolist(),
            )
            cam2color[cam_name_dst] = color_data

        return cam2color

    def build_batch_depth(self) -> dict[str, BatchCameraDataEncoded]:

        cam2depth = {}
        img_idxes = os.listdir(self.camera_dir)
        img_idxes = sorted(img_idxes, key=lambda x: int(x))
        for i in range(self.camera_num):
            cam_name_src = self.cam_names_src[i]
            cam_name_dst = self.cam_names_dst[i]

            intrinsic = self.parameter[cam_name_src]["intrinsic"]
            intrinsic_matrix = self._get_intrinsic_matrix(intrinsic)
            intrinsic_matrix = torch.tensor(
                intrinsic_matrix, dtype=torch.float32
            )

            depth_paths = [
                str(self.camera_dir / img_idx / f"{cam_name_src}_depth.png")
                for img_idx in img_idxes
            ]
            depth_paths = [str(p) for p in depth_paths]

            if Path(depth_paths[0]).exists():
                depth_imgs = load_images_threaded(depth_paths)
                frame_id = f"{cam_name_dst}_camera_depth_optical_frame"
                image_shape_hw = cv2.imread(
                    depth_paths[0], cv2.IMREAD_UNCHANGED
                ).shape[:2]
                intrinsic_matrices = intrinsic_matrix.repeat(
                    len(depth_imgs), 1, 1
                )

                assert len(depth_imgs) == len(self.timestamps_ns), (
                    f"Frame number ({len(depth_imgs)}) and timestamp number "
                    f"({len(self.timestamps_ns)}) do not match in {str(self.camera_dir)}"
                )

                depth_data = BatchCameraDataEncoded(
                    topic=f"/observation/cameras/{cam_name_dst}/depth_image/image_raw",
                    frame_id=frame_id,
                    image_shape=image_shape_hw,
                    intrinsic_matrices=intrinsic_matrices,
                    sensor_data=depth_imgs,
                    format="png",
                    timestamps=self.timestamps_ns.tolist(),
                )
            else:
                depth_data = None

            cam2depth[cam_name_dst] = depth_data

        return cam2depth

    def build_batch_joints(self) -> dict[str, BatchJointsState]:

        ps = self.proprio_states

        # parse state
        state_arm = ps["state"]["joint"]["position"]
        if "effector" in ps["state"]:
            state_gripper = ps["state"]["effector"]["position"]
        else:
            left_abs_gripper = ps["state"]["left_effector"]["position"]
            right_abs_gripper = ps["state"]["right_effector"]["position"]
            if len(left_abs_gripper.shape) == 1:
                left_abs_gripper = np.expand_dims(left_abs_gripper, axis=-1)
                right_abs_gripper = np.expand_dims(right_abs_gripper, axis=-1)
            state_gripper = np.concatenate(
                (left_abs_gripper, right_abs_gripper), axis=-1
            )

        assert (
            state_gripper.shape[1] == 2
        ), f"Expected 2 gripper, got {state_gripper.shape[1]}"
        assert (
            state_arm.shape[1] == 14
        ), f"Expected 14 joints, got {state_arm.shape[1]}"

        # all_abs_head: [N * [head-yaw, head-pitch]]
        state_head = np.array(
            ps["state"]["head"]["position"], dtype=np.float32
        )
        # all_abs_waist: [N * [joint_body_pitch, joint_lift_body]]
        state_body = np.array(
            ps["state"]["waist"]["position"], dtype=np.float32
        )

        left_arm_abs_joint = state_arm[:, : self.joint_dof]
        right_arm_abs_joint = state_arm[:, self.joint_dof :]
        left_arm_abs_gripper = state_gripper[:, : self.gripper_dof]
        right_arm_abs_gripper = state_gripper[:, self.gripper_dof :]
        head_abs_joint = state_head
        try:
            waist_abs_joint = state_body[:, 0:1]
            waist_abs_lift = state_body[:, 1:2]
        except Exception as e:
            waist_abs_joint = np.zeros_like(state_arm)[:, :1]
            waist_abs_lift = np.zeros_like(state_arm)[:, :1]
            logger.error(
                f"{self.episode_dir} has invalid waist state dim: {state_body.shape}, {e}"
            )

        state_joints = np.concatenate(
            [
                left_arm_abs_joint,
                left_arm_abs_gripper,
                right_arm_abs_joint,
                right_arm_abs_gripper,
                head_abs_joint,
                waist_abs_lift,
                waist_abs_joint,
            ],
            axis=-1,
        )

        # parse action
        action_arm = ps["action"]["joint"]["position"]
        if "effector" in ps["action"]:
            action_gripper = ps["action"]["effector"]["position"]
        else:
            left_abs_gripper = ps["action"]["left_effector"]["position"]
            right_abs_gripper = ps["action"]["right_effector"]["position"]
            if len(left_abs_gripper.shape) == 1:
                left_abs_gripper = np.expand_dims(left_abs_gripper, axis=-1)
                right_abs_gripper = np.expand_dims(right_abs_gripper, axis=-1)
            action_gripper = np.concatenate(
                (left_abs_gripper, right_abs_gripper), axis=-1
            )

        assert (
            action_gripper.shape[1] == 2
        ), f"Expected 2 gripper, got {action_gripper.shape[1]}"
        assert (
            action_arm.shape[1] == 14
        ), f"Expected 14 joints, got {action_arm.shape[1]}"

        # all_abs_head: [N * [head-yaw, head-pitch]]
        action_head = np.array(
            ps["action"]["head"]["position"], dtype=np.float32
        )
        # all_abs_waist: [N * [joint_body_pitch, joint_lift_body]]
        action_body = np.array(
            ps["action"]["waist"]["position"], dtype=np.float32
        )

        left_arm_abs_joint = action_arm[:, : self.joint_dof]
        left_arm_abs_gripper = action_gripper[:, : self.gripper_dof]
        right_arm_abs_joint = action_arm[:, self.joint_dof :]
        right_arm_abs_gripper = action_gripper[:, self.gripper_dof :]
        head_abs_joint = action_head
        try:
            waist_abs_joint = action_body[:, 0:1]
            waist_abs_lift = action_body[:, 1:2]
        except Exception as e:
            waist_abs_joint = np.zeros_like(action_arm)[:, :1]
            waist_abs_lift = np.zeros_like(action_arm)[:, :1]
            logger.error(
                f"{self.episode_dir} has invalid waist action dim: {action_body.shape}, {e}"
            )

        action_joints = np.concatenate(
            [
                left_arm_abs_joint,
                left_arm_abs_gripper,
                right_arm_abs_joint,
                right_arm_abs_gripper,
                head_abs_joint,
                waist_abs_lift,
                waist_abs_joint,
            ],
            axis=-1,
        )

        # parse joint names
        joint_names = self._get_joint_names()

        # joint state
        batch_joint_state = BatchJointsState(
            position=torch.tensor(state_joints, dtype=torch.float32),
            names=[joint_name for joint_name in joint_names],
            timestamps=self.timestamps_ns.tolist(),
        )

        # joint action
        batch_joint_action = BatchJointsState(
            position=torch.tensor(action_joints, dtype=torch.float32),
            names=[joint_name for joint_name in joint_names],
            timestamps=self.timestamps_ns.tolist(),
        )

        type2joints = {
            "state": batch_joint_state,
            "action": batch_joint_action,
        }

        return type2joints

    def _get_valid_extrinsic_indexes(self) -> list[int]:
        # align timestamps with extrinsic
        # - 'state.json' len(frames) == 'extrinsic_params_aligned.json' len(extrinsic)
        # - 'aligned_joints.h5' len(timestamps) == 'camera_dir' len(images)
        # - 'state.json' timestamp matches 'camera_dir / time_stamp.json' timestamps

        state = load_json(str(self.state_path))
        time_stamp_paths = [
            str(self.camera_dir / str(img_idx) / f"time_stamp.json")
            for img_idx in range(len(self.timestamps_ns))
        ]
        time_stamps = load_json_threaded(time_stamp_paths)

        ts_to_idx = {}
        for idx, frame in enumerate(state["frames"]):
            ts_to_idx[frame["time_stamp"]] = idx

        valid_time_stamps = [t["head"] for t in time_stamps]
        valid_indexes = []
        for ts in valid_time_stamps:
            if ts in ts_to_idx:
                valid_indexes.append(ts_to_idx[ts])
            else:
                raise ValueError(f"Timestamp {ts} not found in state.json")

        return valid_indexes

    def build_batch_tf_graph(self, type2joints) -> BatchFrameTransformGraph:

        tf_list = []

        # camera pose
        if self.state_path.exists():
            valid_indexes = self._get_valid_extrinsic_indexes()
            for i in range(self.camera_num):
                cam_name_src = self.cam_names_src[i]
                cam_name_dst = self.cam_names_dst[i]

                cam_params = self.parameter[cam_name_src]
                aligned_extrinsic = cam_params["extrinsic"]

                poses = []
                for extinsic in aligned_extrinsic:
                    T_c2b = np.eye(4)
                    T_c2b[:3, :3] = np.array(
                        extinsic['extrinsic']["rotation_matrix"]
                    ).reshape(3, 3)
                    T_c2b[:3, 3:] = np.array(
                        extinsic['extrinsic']["translation_vector"]
                    ).reshape(3, 1)
                    poses.append(T_c2b)
                poses = np.array(poses)  # N x 4 x 4

                t_cam2link = poses[valid_indexes]
                assert t_cam2link.shape[0] == len(
                    self.timestamps_ns
                ), f"Expected {len(self.timestamps_ns)} poses, got {t_cam2link.shape[0]}"

                # NOTE: convert from cam-to-link to cam-to-link-optical
                rotation_x_180 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                t_cam2link = t_cam2link @ rotation_x_180

                xyz, quat_wxyz = self._get_extrinsic_components(t_cam2link)
                tf_cam = BatchFrameTransform(
                    xyz=xyz,
                    quat=quat_wxyz,
                    timestamps=self.timestamps_ns.tolist(),
                    parent_frame_id="base_link",
                    child_frame_id=f"{cam_name_dst}_camera_color_optical_frame",
                )
                tf_list.append(tf_cam)
        else:
            # add cam_to_link, from usda static tf
            parent_link_names = []
            for cam_name_dst in self.cam_names_dst:
                usda_cfg = robot_usda_configs[cam_name_dst]
                xyz, quat_wxyz = get_static_tf_from_usda(usda_cfg)
                # convert to optical frame
                rotation_x_180 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                tf_tmp = BatchFrameTransform(
                    xyz=xyz,
                    quat=quat_wxyz,
                    parent_frame_id=usda_cfg["parent_link"],
                    child_frame_id=f"{cam_name_dst}_camera",
                )
                cam2link = tf_tmp.as_Transform3D_M().get_matrix()
                cam2link_optical = cam2link @ torch.tensor(
                    rotation_x_180, dtype=torch.float32
                )
                # assemble tf
                xyz = cam2link_optical[:, :3, 3]
                quat_wxyz = matrix_to_quaternion(cam2link_optical[:, :3, :3])
                tf_cam = BatchFrameTransform(
                    xyz=xyz.repeat(len(self.timestamps_ns), 1),
                    quat=quat_wxyz.repeat(len(self.timestamps_ns), 1),
                    timestamps=self.timestamps_ns.tolist(),
                    parent_frame_id=usda_cfg["parent_link"],
                    child_frame_id=f"{cam_name_dst}_camera_color_optical_frame",
                )
                tf_list.append(tf_cam)
                parent_link_names.append(usda_cfg["parent_link"])

            # add link_to_base_link, from fk results
            chain = pk.build_chain_from_urdf(self.urdf_content)
            urdf_joint_names = [j.name for j in chain.get_joints()]
            batch_joint_state = type2joints["state"]
            batch_joint_state: BatchJointsState
            fk_input = self._map_to_fk_input(
                urdf_joint_names, batch_joint_state
            )
            link_poses_dict = chain.forward_kinematics(fk_input)

            for link_name, link_tf in link_poses_dict.items():
                if link_name not in parent_link_names:
                    continue

                link_pose = link_tf.get_matrix()
                xyz = link_pose[:, :3, 3]
                wxyz = matrix_to_quaternion(link_pose[:, :3, :3])

                tf = BatchFrameTransform(
                    xyz=xyz.to(torch.float32),
                    quat=wxyz.to(torch.float32),
                    timestamps=self.timestamps_ns.tolist(),
                    parent_frame_id="base_link",
                    child_frame_id=link_name,
                )
                tf_list.append(tf)

        # robot pose
        ps = self.proprio_states
        robot_state_xyz = ps["state"]["robot"]["position"]
        robot_state_xyzw = ps["state"]["robot"]["orientation"]
        robot_state_wxyz = robot_state_xyzw[:, [3, 0, 1, 2]]
        tf_base = BatchFrameTransform(
            xyz=torch.tensor(robot_state_xyz, dtype=torch.float32),
            quat=torch.tensor(robot_state_wxyz, dtype=torch.float32),
            timestamps=self.timestamps_ns.tolist(),
            parent_frame_id="world",
            child_frame_id="base_link",
        )
        tf_list.append(tf_base)

        tf_graph = BatchFrameTransformGraph(tf_list=tf_list)
        return tf_graph

    def build_instruction(self) -> InstructionData:
        meta = self.meta

        name = meta["english_task_name"]
        episode_id = meta["episode_id"]
        description = name.replace("_", " ").capitalize()
        uuid = f"{self.cfg.dataset_name}/{meta['task_id']}/{episode_id}"
        input_action_cfg = meta["label_info"]["action_config"]

        action_config = []
        for cfg in input_action_cfg:
            start_idx = max(0, int(cfg["start_frame"]))
            end_idx = min(int(cfg["end_frame"]), len(self.timestamps_ns) - 1)
            start_ts_ns = self.timestamps_ns[start_idx]
            end_ts_ns = self.timestamps_ns[end_idx]
            action_config.append(
                {
                    "start_ts_ns": int(start_ts_ns),
                    "end_ts_ns": int(end_ts_ns),
                    "action_text": cfg["action_text"],
                    "skill": cfg["skill"],
                    "english_action_text": cfg["english_action_text"],
                }
            )

        instruction = InstructionData(
            name=name,
            json_content={
                "description": description,
                "uuid": uuid,
                "episode_id": episode_id,
                "action_config": action_config,
            },
        )
        return instruction

    @staticmethod
    def get_index_camera_data(
        data: BatchCameraDataEncoded, index: int, allow_none: bool = False
    ) -> BatchCameraDataEncoded:
        if allow_none and data is None:
            return None
        assert data is not None, "Data should not be None"
        assert data.timestamps is not None, "Timestamps should not be None"
        assert (
            data.intrinsic_matrices is not None
        ), "Intrinsic matrices should not be None"
        ret = BatchCameraDataEncoded(
            topic=data.topic,
            frame_id=data.frame_id,
            image_shape=data.image_shape,
            distortion=data.distortion,
            format=data.format,
            sensor_data=[data.sensor_data[index]],
            timestamps=[data.timestamps[index]],
            intrinsic_matrices=data.intrinsic_matrices[index : index + 1],
        )
        return ret

    @staticmethod
    def get_index_transform_graph(
        data: BatchFrameTransformGraph, index: int
    ) -> BatchFrameTransformGraph:
        tf_list = []
        state = data.as_state()
        for tf in state.tf_list:
            assert tf.timestamps is not None, "Timestamps should not be None"
            tf_single = BatchFrameTransform(
                parent_frame_id=tf.parent_frame_id,
                child_frame_id=tf.child_frame_id,
                xyz=tf.xyz[index : index + 1],
                quat=tf.quat[index : index + 1],
                timestamps=[tf.timestamps[index]],
            )
            tf_list.append(tf_single)
        ret = BatchFrameTransformGraph(tf_list=tf_list)
        return ret


class AgibotDigitChallengeEpisodePackaging(EpisodePackaging):

    def __init__(
        self,
        meta: dict,
        urdf_content: str,
        cfg: AgibotDigitChallengePackConfig,
    ):
        self.meta = meta
        self.urdf_content = urdf_content
        self.cfg = cfg

        # check data version
        meta_info = load_json(self.meta["meta_info_path"])
        meta_version = meta_info["version"]
        assert (
            meta_version >= "v0.0.2"
        ), "don't use v0.0.1 data, due to the gripper value means action instead of state."

        self.helper = AgibotDigitChallengeDataHelper(cfg, meta, urdf_content)

    def generate_episode_meta(self) -> EpisodeMeta:
        task_name = self.meta.get("task_name_en", "")
        description = task_name.replace("_", " ").capitalize()
        return EpisodeMeta(
            episode=EpisodeData(),
            robot=RobotData(
                name=self.cfg.robot_name,
                urdf_content=self.urdf_content,
            ),
            task=TaskData(
                name=task_name,
                description=description,
            ),
        )

    def _build_frame_features(self, info: dict, ctx: dict) -> dict:
        feature_type = self.cfg.storage_feature_type

        i = info["raw_frame_index"]

        get_color = self.helper.get_index_camera_data
        get_depth = self.helper.get_index_camera_data
        get_tf = self.helper.get_index_transform_graph

        joint_state = ctx["joints"]["state"]
        joint_action = ctx["joints"]["action"]

        tf_graph_i = get_tf(ctx["tf_graph"], i)

        cam2color = ctx["cam2color"]
        cam2depth = ctx["cam2depth"]

        color_middle = get_color(cam2color["middle"], i)
        color_left = get_color(cam2color["left"], i)
        color_right = get_color(cam2color["right"], i)
        color_middle.pose = tf_graph_i.get_tf("world", "middle_camera_color_optical_frame")  # type: ignore
        color_left.pose = tf_graph_i.get_tf("world", "left_camera_color_optical_frame")  # type: ignore
        color_right.pose = tf_graph_i.get_tf("world", "right_camera_color_optical_frame")  # type: ignore

        depth_middle = get_depth(cam2depth["middle"], i, allow_none=True)
        depth_left = get_depth(cam2depth["left"], i, allow_none=True)
        depth_right = get_depth(cam2depth["right"], i, allow_none=True)

        if feature_type == "raw":
            features = {
                # index
                "raw_frame_index": i,
                # joints
                "robot/joints/state": joint_state[i],
                "robot/joints/action": joint_action[i],
                # cameras
                "cameras/middle/color": color_middle,
                "cameras/left/color": color_left,
                "cameras/right/color": color_right,
                "cameras/middle/depth": depth_middle,
                "cameras/left/depth": depth_left,
                "cameras/right/depth": depth_right,
                # tf
                "tf_graph": tf_graph_i,
            }

        elif feature_type == "compact":
            features = {
                # index
                "raw_frame_index": i,
                # joints - episode level
                "joint_state": ctx["joint_state"],
                "joint_action": ctx["joint_action"],
                # cameras
                "cameras/middle/color": color_middle,
                "cameras/left/color": color_left,
                "cameras/right/color": color_right,
                "cameras/middle/depth": depth_middle,
                "cameras/left/depth": depth_left,
                "cameras/right/depth": depth_right,
                # tf
                "tf_graph": tf_graph_i,
            }

        elif feature_type == "lite":
            lite_meta = info["lite_meta"]
            frame_index_in_chunk = lite_meta["frame_index_in_chunk"]
            chunk_indices = lite_meta["chunk_indices"]

            features = {
                # index
                "raw_frame_index": i,
                "frame_index_in_chunk": frame_index_in_chunk,
                "joint_raw_frame_index": list(chunk_indices),
                # joints - chunk level
                "joint_state": ctx["joint_state"][chunk_indices],
                "joint_action": ctx["joint_action"][chunk_indices],
                # cameras
                "cameras/middle_color": None,
                "cameras/left_color": None,
                "cameras/right_color": None,
                "cameras/middle_depth": None,
                "cameras/left_depth": None,
                "cameras/right_depth": None,
                # intrinsic
                "camera_intrinsics/middle": None,
                "camera_intrinsics/left": None,
                "camera_intrinsics/right": None,
                # tf
                "tf_graph": tf_graph_i,
                # text & uuid
                "text": "",
                "uuid": "",
            }

            # update intrinsic
            for camera_name in self.helper.cam_names_dst:
                intrinsic = (
                    cam2color[camera_name]
                    .intrinsic_matrices[0]
                    .numpy()
                    .flatten()
                )
                # frame level storage
                features[f"camera_intrinsics/{camera_name}"] = (
                    intrinsic.tolist()
                )

            # use simple type image data
            name_color = {
                "middle": color_middle,
                "left": color_left,
                "right": color_right,
            }
            name_depth = {
                "middle": depth_middle,
                "left": depth_left,
                "right": depth_right,
            }
            for camera_name in self.helper.cam_names_dst:
                color_key = f"cameras/{camera_name}_color"
                depth_key = f"cameras/{camera_name}_depth"
                features[color_key] = name_color[camera_name].sensor_data
                if name_depth[camera_name] is not None:
                    features[depth_key] = name_depth[camera_name].sensor_data
                else:
                    features[depth_key] = None

            # text & uuid
            json_content = ctx["instruction"].json_content or {}
            action_config = json_content.get("action_config", [])
            action_text = [c["english_action_text"] for c in action_config]
            text = ";".join(action_text)
            features["text"] = text
            features["uuid"] = json_content.get("uuid", "")

        else:
            raise ValueError(
                f"Unknown storage_feature_type: {self.cfg.storage_feature_type}"
            )

        return features

    def generate_frames(self):
        logger.info(f"Processing episode: {self.meta['episode_id']}")

        helper = self.helper
        ctx = {
            "cam2color": helper.build_batch_color(),
            "cam2depth": helper.build_batch_depth(),
            "joints": helper.build_batch_joints(),
            "instruction": helper.build_instruction(),
            "total_frames": len(helper.timestamps_ns),
        }
        ctx["tf_graph"] = helper.build_batch_tf_graph(ctx["joints"])

        if ctx["total_frames"] == 0:
            logger.warning(
                f"Episode {self.meta['episode_id']} has 0 frames after processing."
            )
            return

        # prepare mask
        generate_static_mask(ctx["joints"]["state"].position, threshold=1e-4)
        static_mask = generate_static_mask(
            ctx["joints"]["state"].position,
            threshold=self.cfg.static_threshold,
        )  # mask is True for static frames
        logger.info(
            f"Static frames: {static_mask.sum().item()} / {ctx['total_frames']}"
        )
        valid_indices = [
            raw_idx
            for raw_idx, is_static in enumerate(static_mask)
            if not is_static
        ]
        raw_to_valid_pos = {
            raw_idx: pos for pos, raw_idx in enumerate(valid_indices)
        }

        # filter
        filtered_frame_infos = []
        for i in range(ctx["total_frames"]):
            # filter static frames
            if self.cfg.skip_static_frames and static_mask[i]:
                continue
            frame_info = {
                "raw_frame_index": i,
                "timestamp_ns": helper.timestamps_ns[i],
            }
            if self.cfg.storage_feature_type == "lite":
                cur_pos = raw_to_valid_pos[i]
                start = max(0, cur_pos - (self.cfg.lite_mode_hist_step - 1))
                end = min(
                    len(valid_indices),
                    cur_pos + self.cfg.lite_mode_pred_step + 1,
                )
                frame_info["lite_meta"] = {
                    "chunk_indices": valid_indices[start:end],
                    "frame_index_in_chunk": cur_pos - start,
                }
            filtered_frame_infos.append(frame_info)
        logger.info(
            f"Total frames after filtering: {len(filtered_frame_infos)}"
        )

        # writer
        np_state = ctx["joints"]["state"].position.detach().cpu().numpy().astype("float32")  # fmt: skip
        np_action = ctx["joints"]["action"].position.detach().cpu().numpy().astype("float32")  # fmt: skip
        ctx.update(
            {
                "joint_state": np_state,
                "joint_action": np_action,
            }
        )
        for info in filtered_frame_infos:
            features = self._build_frame_features(info, ctx)
            yield DataFrame(
                features=features,
                instruction=ctx["instruction"],
                timestamp_ns_max=info["timestamp_ns"],
                timestamp_ns_min=info["timestamp_ns"],
            )

        return


class AgibotDigitChallengeRODataPacker(BaseRODataPacker):

    cfg: AgibotDigitChallengePackConfig

    def __init__(self, cfg: AgibotDigitChallengePackConfig):
        super().__init__(cfg)
        self.urdf_content = load_txt(cfg.urdf_path)

    def get_dataset_features(self):
        feature_type = self.cfg.storage_feature_type.lower()
        d = self.cfg.joint_dimensions

        if feature_type == "raw":
            features = {
                # index
                "raw_frame_index": Value("int32"),
                # joints
                "robot/joints/state": BatchJointsStateFeature(),
                "robot/joints/action": BatchJointsStateFeature(),
                # cameras
                "cameras/middle/color": BatchCameraDataEncodedFeature(),
                "cameras/left/color": BatchCameraDataEncodedFeature(),
                "cameras/right/color": BatchCameraDataEncodedFeature(),
                "cameras/middle/depth": BatchCameraDataEncodedFeature(),
                "cameras/left/depth": BatchCameraDataEncodedFeature(),
                "cameras/right/depth": BatchCameraDataEncodedFeature(),
                # tf
                "tf_graph": BatchFrameTransformGraphFeature(),
            }

        elif feature_type == "compact":
            features = {
                # index
                "raw_frame_index": Value("int32"),
                # joints
                "joint_state": Array2D((None, d), "float32"),
                "joint_action": Array2D((None, d), "float32"),
                # cameras
                "cameras/middle_color": BatchCameraDataEncodedFeature(),
                "cameras/left_color": BatchCameraDataEncodedFeature(),
                "cameras/right_color": BatchCameraDataEncodedFeature(),
                "cameras/middle_depth": BatchCameraDataEncodedFeature(),
                "cameras/left_depth": BatchCameraDataEncodedFeature(),
                "cameras/right_depth": BatchCameraDataEncodedFeature(),
                # tf
                "tf_graph": BatchFrameTransformGraphFeature(),
            }

        elif feature_type == "lite":
            features = {
                # index
                "raw_frame_index": Value("int32"),
                "frame_index_in_chunk": Value("int32"),
                # joints
                "joint_state": Array2D((None, d), "float32"),
                "joint_action": Array2D((None, d), "float32"),
                "joint_raw_frame_index": Sequence(Value("int32")),
                # cameras
                "cameras/middle_color": Sequence(Value("binary")),
                "cameras/left_color": Sequence(Value("binary")),
                "cameras/right_color": Sequence(Value("binary")),
                "cameras/middle_depth": Sequence(Value("binary")),
                "cameras/left_depth": Sequence(Value("binary")),
                "cameras/right_depth": Sequence(Value("binary")),
                "camera_intrinsics/middle": Sequence(Value("float32"), length=9),  # fmt: skip
                "camera_intrinsics/left": Sequence(Value("float32"), length=9),
                "camera_intrinsics/right": Sequence(Value("float32"), length=9),  # fmt: skip
                "text": Value("string"),
                "uuid": Value("string"),
                # tf
                "tf_graph": BatchFrameTransformGraphFeature(),
            }

        else:
            raise ValueError(
                f"Unknown storage_feature_type: {feature_type!r}. "
                f"Supported values: 'raw', 'compact', 'lite'"
            )

        return hg_datasets.Features(features)

    def _scan_all_episode_metas(self) -> list[dict]:
        task_train_paths = Path(self.cfg.input_dir).glob("*/task_train.json")
        task_train_paths = sorted(task_train_paths)

        episode_to_meta = {}

        for task_path in tqdm(task_train_paths, desc="Loading task infos"):
            task_dir = task_path.parent
            task_name_en = task_dir.name

            task_infos = load_json(str(task_path))

            for info in task_infos:
                task_id = str(info["task_id"])
                job_id = str(info["job_id"])
                sn_code = info["sn_code"]
                episode_id = str(info["episode_id"])

                ep_dir = task_dir / task_id / job_id / sn_code / episode_id
                meta_path = ep_dir / "meta_info.json"

                episode_key = f"{task_id}/{episode_id}"
                meta = info.copy()
                meta.update(
                    {
                        "task_id": task_id,
                        "job_id": job_id,
                        "episode_id": episode_id,
                        "task_name_en": task_name_en,
                        "meta_info_path": str(meta_path),
                    }
                )

                episode_to_meta[episode_key] = meta

        sorted_metas = sorted(
            episode_to_meta.values(),
            key=lambda x: (x["task_id"], x["episode_id"]),
        )

        return sorted_metas

    def _filter_metas(self, metas: list[dict]) -> list[dict]:

        # filter by task name
        total_before = len(metas)
        if self.cfg.task_name:
            metas = [
                m for m in metas if m["task_name_en"] == self.cfg.task_name
            ]
            logger.info(
                f"Filtered metas by task_name='{self.cfg.task_name}': "
                f"{len(metas)} / {total_before}"
            )

        # filter by split
        total_before = len(metas)
        if self.cfg.split == "train":
            metas = [m for i, m in enumerate(metas) if (i % 20) != 0]
            logger.info(
                f"Filtered metas by split='{self.cfg.split}': "
                f"{len(metas)} / {total_before}"
            )
        elif self.cfg.split == "val":
            metas = [m for i, m in enumerate(metas) if (i % 20) == 0]
            logger.info(
                f"Filtered metas by split='{self.cfg.split}': "
                f"{len(metas)} / {total_before}"
            )
        else:
            logger.info(f"No split filtering applied.")

        return metas

    def collect_all_metas(self) -> list[dict]:

        if self.cfg.cached_meta_path:
            cached_path = Path(self.cfg.cached_meta_path)
        else:
            input_dir = Path(self.cfg.input_dir)
            cached_path = input_dir / "cached_episode_metas.json"

        if cached_path.is_file():
            logger.info(f"Loading cached metas from {cached_path}")
            metas = load_json(str(cached_path))
        else:
            logger.info("No valid cache found, scanning input directory...")
            metas = self._scan_all_episode_metas()
            # save cached metas
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(str(cached_path), metas)

        metas = self._filter_metas(metas)
        logger.info(f"Final collected metas count: {len(metas)}")

        return metas

    def build_episode(self, meta: dict) -> EpisodePackaging:
        episode = AgibotDigitChallengeEpisodePackaging(
            meta, self.urdf_content, self.cfg
        )
        return episode


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d - %(message)s",
        force=True,
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--urdf_path", type=str)
    parser.add_argument("--robot_name", type=str, default="G1_omnipicker")
    parser.add_argument("--force_overwrite", action="store_true")
    parser.add_argument("--cached_meta_path", type=str, default="")
    parser.add_argument("--task_name", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--storage_feature_type", type=str, default="lite")
    # slice parameters
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_idx", type=int, default=0)

    args = parser.parse_args()
    pack_cfg = AgibotDigitChallengePackConfig(
        dataset_name="AgibotDigitChallenge",
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        robot_name=args.robot_name,
        urdf_path=args.urdf_path,
        force_overwrite=args.force_overwrite,
        num_jobs=args.num_jobs,
        job_idx=args.job_idx,
        cached_meta_path=args.cached_meta_path,
        task_name=args.task_name,
        split=args.split,
        storage_feature_type=args.storage_feature_type,
    )

    packer = AgibotDigitChallengeRODataPacker(pack_cfg)
    packer.pack()
