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

import copy
import glob
import json
import logging
import multiprocessing
import os
import subprocess

import cv2
import numpy as np
import pytorch_kinematics as pk
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.horizon_manipulation.tools.utils import (
    format_time,
    get_frequency,
    pose_to_mat,
    sample,
)
from robo_orchard_lab.dataset.horizon_manipulation.utils import (
    decode_depth,
    decode_img,
    depth_visualize,
)

logger = logging.getLogger(__name__)


def get_h264_encoder() -> str:
    """Return an available ffmpeg H.264 encoder."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    encoders = result.stdout
    for encoder in ("libx264", "libopenh264", "h264_v4l2m2m"):
        if encoder in encoders:
            return encoder
    raise RuntimeError("No supported H.264 encoder found in ffmpeg")


class FfmpegVideoWriter:
    """Stream raw BGR frames to ffmpeg and encode them as mp4."""

    def __init__(self, video_file: str, frame_size, fps: int):
        width, height = frame_size
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            get_h264_encoder(),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            video_file,
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.stdin = self.process.stdin

    def write(self, frame: np.ndarray) -> None:
        if self.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        self.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.stdin is not None and not self.stdin.closed:
            self.stdin.close()
        returncode = self.process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with non-zero status: {returncode}"
            )


class PiperMcapChecker:
    """Abstact class of Packing data into target packtype.

    Packer is the recommended base class for being inherited.
    Focus on packer env, read and write.

    Subclass need to override :py:func:`pack_data`.

    Args:
        max_data_num (int): Num of data for packing.
        pack_type (str): The target pack type.
        num_workers (int): Num workers for reading original data.
            while num_workers <= 0 means pack by single process.
            num_workers >= 1 mean pack by num_workers process.
        **kwargs (dict): kwargs for pack type.
    """

    def __init__(
        self,
        input_path,
        output_path,
        urdf,
        calibration_dict=None,
        task_names=None,
        user_names=None,
        static_threshold: float = 1e-3,
        head_time_to_filter: float = 1e8,
        tile_time_to_filter: float = None,
        date_prefix: str = None,
        num_workers: int = 4,
        vis_interval: int = 3,
        vis_fps: int = 150,
        vis_resize_scale: float = 4,
        vis_depth: bool = True,
        **kwargs,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.calibration_dict = self.calibration_process(calibration_dict)
        self.task_names = task_names
        self.user_names = user_names
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read())
        self.static_threshold = static_threshold
        self.head_time_to_filter = head_time_to_filter
        self.tile_time_to_filter = tile_time_to_filter
        if date_prefix is not None:
            self.date_prefix = date_prefix.strip().split(",")
        else:
            self.date_prefix = date_prefix
        self.episodes = self.input_path_handler(input_path)
        self.num_workers = num_workers
        self.vis_interval = vis_interval
        self.vis_fps = vis_fps
        self.vis_resize_scale = vis_resize_scale
        self.vis_depth = vis_depth

    def calibration_process(self, calibration_dict):
        if calibration_dict is None:
            return None
        for camera, calib in calibration_dict.items():
            calibration_dict[camera] = np.linalg.inv(pose_to_mat(calib))
        return calibration_dict

    def input_path_handler(self, input_paths):
        episodes = []
        input_paths = input_paths.strip().split(",")
        logger.info(f"input_paths: {input_paths}")
        for input_path in input_paths:
            for user in os.listdir(input_path):
                if self.user_names is not None and user not in self.user_names:
                    continue

                user_path = os.path.join(input_path, user)
                for task in os.listdir(user_path):
                    if (
                        self.task_names is not None
                        and task not in self.task_names
                    ):
                        continue
                    task_path = os.path.join(user_path, task)
                    for ep in os.listdir(task_path):
                        if not ep.startswith("episode_"):
                            logger.warning(f"invalid episode dir name {ep}")
                            continue
                        path = os.path.join(task_path, ep)
                        time = ep.replace("episode_", "")

                        if self.date_prefix is not None and not any(
                            [time.startswith(x) for x in self.date_prefix]
                        ):
                            continue
                        episodes.append([path, user, task, time])
        episodes.sort()
        logger.info(f"number of valid episodes: {len(episodes)}")
        return episodes

    def forward_kinematics(self, left_joint, right_joint):
        joint = np.zeros([left_joint.shape[0], 16])
        if left_joint is not None:
            joint[:, :6] = left_joint[:, :6]
            joint[:, 6:7] = left_joint[:, 6:7] / 2
            joint[:, 7:8] = -left_joint[:, 6:7] / 2
        if right_joint is not None:
            joint[:, 8:14] = right_joint[:, :6]
            joint[:, 14:15] = right_joint[:, 6:7] / 2
            joint[:, 15:16] = -right_joint[:, 6:7] / 2
        link_poses_dict = self.chain.forward_kinematics(joint)
        return link_poses_dict

    def parse(self, ep_id):
        cameras = ["left", "middle", "right"]
        image_topics = [
            f"/observation/cameras/{x}/color_image/image_raw" for x in cameras
        ]
        image_intrinsic_topics = [
            f"/observation/cameras/{x}/color_image/camera_info"
            for x in cameras
        ]
        depth_topics = [
            f"/observation/cameras/{x}/depth_image/image_raw" for x in cameras
        ]
        depth_intrinsic_topics = [
            f"/observation/cameras/{x}/depth_image/camera_info"
            for x in cameras
        ]
        image_extrinsic_topic = ["/tf_static"]
        left_joint_topic = "/observation/robot_state/left/joint"
        right_joint_topic = "/observation/robot_state/right/joint"
        left_master_joint_topic = "/observation/robot_state/left_master/joint"
        right_master_joint_topic = (
            "/observation/robot_state/right_master/joint"
        )
        left_ee_topic = "/observation/robot_state/left/end_pose"
        right_ee_topic = "/observation/robot_state/right/end_pose"

        frame_id_extrinsic_map = {
            "middle_camera_color_optical_frame/left_base_link": "middle",
            "left_camera_color_optical_frame/left_end_effector": "left",
            "right_camera_color_optical_frame/right_end_effector": "right",
        }

        all_useful_topics = (
            image_topics
            + image_intrinsic_topics
            + depth_topics
            + depth_intrinsic_topics
            + image_extrinsic_topic
            + [
                left_joint_topic,
                right_joint_topic,
                left_ee_topic,
                right_ee_topic,
            ]
            + [left_master_joint_topic, right_master_joint_topic]
        )

        ep = self.episodes[ep_id]
        path, user, task_name, time = ep

        uuid = f"{task_name}/{user}/{time}"
        mcap = glob.glob(f"{path}/episode_*_0.mcap")
        assert len(mcap) == 1
        mcap = mcap[0]
        meta = json.load(open(os.path.join(path, "episode_meta.json"), "r"))
        assert user == meta["user_name"]
        assert task_name == meta["task_name"]

        image_extrinsic = {}
        images = {}
        for t in image_topics:
            images[t] = {
                "data": [],
                "time": [],
                "intrinsic": None,
            }
        depths = {}
        for t in depth_topics:
            depths[t] = {
                "data": [],
                "time": [],
                "intrinsic": None,
            }
        joints = {}
        for t in [
            left_joint_topic,
            right_joint_topic,
            left_master_joint_topic,
            right_master_joint_topic,
        ]:
            joints[t] = {
                "position": [],
                "velocity": [],
                "effort": [],
                "time": [],
            }
        ee_poses = {}
        for t in [left_ee_topic, right_ee_topic]:
            ee_poses[t] = {
                "pose": [],
                "time": [],
            }

        reader = make_reader(
            open(mcap, "rb"), decoder_factories=[DecoderFactory()]
        )

        for (
            _schema,
            channel,
            _message,
            ros_msg,
        ) in reader.iter_decoded_messages(
            topics=all_useful_topics, log_time_order=True
        ):
            if hasattr(ros_msg, "header"):
                time = (
                    ros_msg.header.stamp.sec,
                    ros_msg.header.stamp.nanosec,
                )
            topic = channel.topic
            if topic in images:
                images[topic]["data"].append(ros_msg.data)
                images[topic]["time"].append(time)
            elif topic in depths:
                depths[topic]["data"].append(ros_msg.data)
                depths[topic]["time"].append(time)
            elif topic in joints:
                joints[topic]["position"].append(ros_msg.position)
                joints[topic]["velocity"].append(ros_msg.velocity)
                joints[topic]["effort"].append(ros_msg.effort)
                joints[topic]["time"].append(time)
            elif topic in ee_poses:
                pos = ros_msg.pose.position
                rot = ros_msg.pose.orientation
                x, y, z = pos.x, pos.y, pos.z
                qx, qy, qz, qw = rot.x, rot.y, rot.z, rot.w
                ee_poses[topic]["pose"].append([x, y, z, qx, qy, qz, qw])
                ee_poses[topic]["time"].append(time)
            elif topic in image_intrinsic_topics:
                img_topic = image_topics[image_intrinsic_topics.index(topic)]
                images[img_topic]["intrinsic"] = np.array(ros_msg.p).reshape(
                    3, 4
                )
            elif topic in depth_intrinsic_topics:
                dpt_topic = depth_topics[depth_intrinsic_topics.index(topic)]
                depths[dpt_topic]["intrinsic"] = np.array(ros_msg.p).reshape(
                    3, 4
                )
            elif topic in image_extrinsic_topic:
                for tf in ros_msg.transforms:
                    frame_id = f"{tf.child_frame_id}/{tf.header.frame_id}"
                    if frame_id in frame_id_extrinsic_map:
                        camera_name = frame_id_extrinsic_map[frame_id]
                        extrinsic = np.linalg.inv(pose_to_mat(tf.transform))
                        image_extrinsic[camera_name] = extrinsic

        for data in [images, depths, joints, ee_poses]:
            for t in data:
                data[t]["time"] = format_time(data[t]["time"])

        base_time = images[image_topics[0]]["time"]
        num_steps = len(base_time)
        for t in images:
            images[t]["freq"] = get_frequency(
                images[t]["time"], uuid + " | " + t
            )
            if t == image_topics[0]:
                continue
            images[t]["time"], (images[t]["data"],) = sample(
                base_time,
                images[t]["time"],
                [images[t]["data"]],
                prefix=uuid + " | " + t,
            )
        for t in depths:
            depths[t]["freq"] = get_frequency(
                depths[t]["time"], uuid + " | " + t
            )
            depths[t]["time"], (depths[t]["data"],) = sample(
                base_time,
                depths[t]["time"],
                [depths[t]["data"]],
                prefix=uuid + " | " + t,
            )
        for t in joints:
            joints[t]["freq"] = get_frequency(
                joints[t]["time"], uuid + " | " + t
            )
            (
                joints[t]["time"],
                (
                    joints[t]["position"],
                    joints[t]["velocity"],
                    joints[t]["effort"],
                ),
            ) = sample(
                base_time,
                joints[t]["time"],
                [
                    joints[t]["position"],
                    joints[t]["velocity"],
                    joints[t]["effort"],
                ],
                prefix=uuid + " | " + t,
            )
            joints[t]["position"] = np.array(joints[t]["position"])

        poses = self.forward_kinematics(
            joints[left_joint_topic]["position"],
            joints[right_joint_topic]["position"],
        )
        left_ee_pose = poses["left_gripper_base"].get_matrix().numpy()
        right_ee_pose = poses["right_gripper_base"].get_matrix().numpy()

        for t in ee_poses:
            ee_poses[t]["freq"] = get_frequency(
                ee_poses[t]["time"], uuid + " | " + t
            )
            ee_poses[t]["time"], (ee_poses[t]["pose"],) = sample(
                base_time,
                ee_poses[t]["time"],
                [ee_poses[t]["pose"]],
                prefix=uuid + " | " + t,
            )

        if self.calibration_dict is not None:
            image_extrinsic = copy.deepcopy(self.calibration_dict)
        extrinsic = {
            "left": image_extrinsic["left"] @ np.linalg.inv(left_ee_pose),
            "right": image_extrinsic["right"] @ np.linalg.inv(right_ee_pose),
            "middle": np.copy(image_extrinsic["middle"]),
        }

        intrinsic = {}
        for cam, t in zip(cameras, image_topics, strict=False):
            intrinsic[cam] = images[t]["intrinsic"]
        joint_positions = np.concatenate(
            [
                np.array(joints[left_joint_topic]["position"]),
                np.array(joints[right_joint_topic]["position"]),
            ],
            axis=-1,
        )
        joint_velocity = np.concatenate(
            [
                np.array(joints[left_joint_topic]["velocity"]),
                np.array(joints[right_joint_topic]["velocity"]),
            ],
            axis=-1,
        )
        joint_effort = np.concatenate(
            [
                np.array(joints[left_joint_topic]["effort"]),
                np.array(joints[right_joint_topic]["effort"]),
            ],
            axis=-1,
        )

        master_joint_positions = np.concatenate(
            [
                np.array(joints[left_master_joint_topic]["position"]),
                np.array(joints[right_master_joint_topic]["position"]),
            ],
            axis=-1,
        )
        master_joint_velocity = np.concatenate(
            [
                np.array(joints[left_master_joint_topic]["velocity"]),
                np.array(joints[right_master_joint_topic]["velocity"]),
            ],
            axis=-1,
        )
        master_joint_effort = np.concatenate(
            [
                np.array(joints[left_master_joint_topic]["effort"]),
                np.array(joints[right_master_joint_topic]["effort"]),
            ],
            axis=-1,
        )

        ee_poses = np.concatenate(
            [
                left_ee_pose[:, :3, 3],
                Rotation.from_matrix(left_ee_pose[:, :3, :3]).as_quat(
                    scalar_first=False
                ),
                right_ee_pose[:, :3, 3],
                Rotation.from_matrix(right_ee_pose[:, :3, :3]).as_quat(
                    scalar_first=False
                ),
            ],
            axis=1,
        ).reshape(-1, 2, 7)

        # ==================== filter
        if self.static_threshold is not None and self.static_threshold > 0:
            static_mask = np.ones(joint_positions.shape[0], dtype=bool)
            static_mask[1:] = np.any(
                np.abs(np.diff(joint_positions, axis=0)) > 1e-3, axis=1
            )

            time_mask = np.zeros(static_mask.shape[0], bool)
            if (
                self.head_time_to_filter is not None
                and self.head_time_to_filter > 0
            ):
                head_time = base_time - base_time[0]
                time_mask = np.logical_or(
                    time_mask, head_time < self.head_time_to_filter
                )
            if (
                self.tile_time_to_filter is not None
                and self.tile_time_to_filter > 0
            ):
                tile_time = -base_time + base_time[-1]
                time_mask = np.logical_or(
                    time_mask, tile_time < self.tile_time_to_filter
                )
            static_mask = np.logical_or(static_mask, np.logical_not(time_mask))
            joint_positions = joint_positions[static_mask]
            joint_velocity = joint_velocity[static_mask]
            joint_effort = joint_effort[static_mask]
            master_joint_positions = master_joint_positions[static_mask]
            master_joint_velocity = master_joint_velocity[static_mask]
            master_joint_effort = master_joint_effort[static_mask]
            ee_poses = ee_poses[static_mask]
            logger.info(
                f"{uuid} | all steps: {num_steps}, "
                f"non static steps: {joint_positions.shape[0]}"
            )
            num_steps = joint_positions.shape[0]

            extrinsic["left"] = extrinsic["left"][static_mask]
            extrinsic["right"] = extrinsic["right"][static_mask]
            base_time = base_time[static_mask]
            for t in image_topics:
                images[t]["data"] = [
                    x
                    for i, x in enumerate(images[t]["data"])
                    if static_mask[i]
                ]
            for t in depth_topics:
                depths[t]["data"] = [
                    x
                    for i, x in enumerate(depths[t]["data"])
                    if static_mask[i]
                ]
        # ==================== filter finish

        return dict(
            cam_names=cameras,
            uuid=uuid,
            images=images,
            depths=depths,
            joint_positions=joint_positions,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            master_joint_positions=master_joint_positions,
            master_joint_velocity=master_joint_velocity,
            master_joint_effort=master_joint_effort,
            ee_poses=ee_poses,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            meta=meta,
        )

    def draw_ee_pose(self, img, intrinsic, extrinsic, ee_poses):
        rot = Rotation.from_quat(
            ee_poses[:, 3:], scalar_first=True
        ).as_matrix()
        trans = ee_poses[:, :3]
        axis_length = 0.1
        points = np.float32(
            [
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length],
                [0, 0, 0],
            ]
        )
        # npose, 1, 3, 3 @ [4, 3, 3] + [npose, 1, 3]
        points = (rot[:, None] @ points[..., None]).squeeze(-1)
        points += trans[:, None]  # npose, 4, 3
        projection_mat = intrinsic @ extrinsic

        pts_2d = points @ projection_mat[:3, :3].T
        pts_2d = pts_2d + projection_mat[:3, 3]
        depth = pts_2d[..., 2]
        pts_2d = pts_2d[..., :2] / depth[..., None]
        pts_2d = pts_2d.astype(np.int32)

        for pose_idx in range(len(ee_poses)):
            for i in range(3):
                if depth[pose_idx, i] < 0.02:
                    continue
                cv2.circle(
                    img,
                    (pts_2d[pose_idx, i, 0], pts_2d[pose_idx, i, 1]),
                    6,
                    (0, 0, 255),
                    -1,
                )
                if i == 3:
                    continue
                color = [0, 0, 0]
                color[i] = 255
                cv2.line(
                    img,
                    (pts_2d[pose_idx, i, 0], pts_2d[pose_idx, i, 1]),
                    (pts_2d[pose_idx, 3, 0], pts_2d[pose_idx, 3, 1]),
                    tuple(color),
                    3,
                )
        return img

    def decode_and_visualize(self, data):
        video_writer = None
        uuid = data["uuid"]
        os.makedirs(self.output_path, exist_ok=True)
        file = os.path.join(
            self.output_path, f"{uuid.replace(os.sep, '-')}.mp4"
        )
        num_steps = len(data["joint_positions"])
        for i in range(0, num_steps, self.vis_interval):
            imgs = []
            depths = []
            ee_poses = data["ee_poses"][i]
            for cam in data["cam_names"]:
                image_topic = (
                    f"/observation/cameras/{cam}/color_image/image_raw"
                )
                depth_topic = (
                    f"/observation/cameras/{cam}/depth_image/image_raw"
                )
                extrinsic = data["extrinsic"][cam]
                if extrinsic.ndim == 3:
                    extrinsic = extrinsic[i]
                intrinsic = data["intrinsic"][cam]

                img = decode_img(data["images"][image_topic]["data"][i])
                img = self.draw_ee_pose(img, intrinsic, extrinsic, ee_poses)
                depth = decode_depth(
                    data["depths"][depth_topic]["data"][i], 1000
                )
                depth = depth_visualize(depth)

                imgs.append(img)
                depths.append(depth)

            shapes = [im.shape[:2] for im in imgs + depths]
            if not all(s == shapes[0] for s in shapes):
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                if i == 0:
                    logger.warning(
                        f"{uuid} | inconsistent image shapes detected, "
                        f"resizing all images to ({max_h},{max_w})"
                    )

                def _resize(im, size):
                    return cv2.resize(im, size, interpolation=cv2.INTER_AREA)

                imgs = [_resize(im, (max_w, max_h)) for im in imgs]
                depths = [_resize(im, (max_w, max_h)) for im in depths]

            if self.vis_depth:
                vis_img = np.concatenate(
                    [
                        np.concatenate(imgs, axis=1),
                        np.concatenate(depths, axis=1),
                    ],
                    axis=0,
                )
            else:
                vis_img = np.concatenate(imgs, axis=1)

            if self.vis_resize_scale is not None:
                size = (
                    vis_img.shape[1] // self.vis_resize_scale,
                    vis_img.shape[0] // self.vis_resize_scale,
                )
                vis_img = cv2.resize(
                    vis_img, size, interpolation=cv2.INTER_AREA
                )

            cv2.putText(
                vis_img,
                uuid,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(1 / self.vis_resize_scale, 0.8),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                vis_img,
                data["meta"]["instruction"],
                (10, vis_img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(1 / self.vis_resize_scale, 0.8),
                (0, 0, 255),
                2,
            )

            if video_writer is None:
                video_writer = FfmpegVideoWriter(
                    file,
                    vis_img.shape[:2][::-1],
                    self.vis_fps // self.vis_interval,
                )
            video_writer.write(vis_img)
        if video_writer is not None:
            video_writer.release()
        return file

    def handler(self, ep_id):
        logger.info(f"start process [{ep_id + 1}/{len(self.episodes)}]")
        try:
            episode_data = self.parse(ep_id)
            uuid = episode_data["uuid"]
            num_steps = len(episode_data["joint_positions"])
            video_file = self.decode_and_visualize(episode_data)
        except Exception as e:
            uuid = self.episodes[ep_id][0]
            logger.exception(
                f"failed to process episode {ep_id} with error: {e}"
            )
            return None
        else:
            logger.info(
                f"finish process [{ep_id + 1}/{len(self.episodes)}] {uuid}, "
                f"num_steps:{num_steps} \n"
            )
        return dict(
            uuid=uuid,
            num_steps=num_steps,
            video_file=video_file,
        )

    def concat_videos(self, video_files):
        list_file = os.path.join(self.output_path, "video_files.txt")
        with open(list_file, "w") as f:
            for file in video_files:
                f.write(f"file '{os.path.abspath(file)}'\n")
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            os.path.join(self.output_path, "concat_videos.mp4"),
        ]
        subprocess.run(cmd, check=True)

    def __call__(self):
        if self.num_workers > 1:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                results = pool.map(
                    self.handler, list(range(len(self.episodes)))
                )
        else:
            results = []
            for i in range(len(self.episodes)):
                results.append(self.handler(i))

        results = [x for x in results if x is not None]
        results.sort(key=lambda x: x["uuid"])
        total_steps = sum([x["num_steps"] for x in results])

        logger.info(
            f"total steps: {total_steps}, "
            f"total times: {total_steps / 108000:.3f}h"
        )
        self.concat_videos([x["video_file"] for x in results])


if __name__ == "__main__":
    import argparse

    from robo_orchard_lab.utils import log_basic_config

    log_basic_config(
        format="%(asctime)s %(levelname)s-%(lineno)d: %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--user_names", type=str, default=None)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--date_prefix", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--embodiedment_meta_file", type=str, default=None)
    parser.add_argument("--embodiedment", type=str, default=None)
    parser.add_argument("--use_extra_calibration", action="store_true")
    args = parser.parse_args()

    if args.task_names is None:
        task_names = None
    else:
        task_names = args.task_names.split(",")

    calibration_dict = None
    if args.embodiedment_meta_file is not None:
        with open(args.embodiedment_meta_file, "r") as f:
            embodiedment_meta = json.load(f)
        embodiedment_meta = embodiedment_meta[args.embodiedment]
        if args.use_extra_calibration:
            calibration_dict = embodiedment_meta["calibration"]
            logger.info(f"use extra calibration: {calibration_dict}")
        if args.urdf is None:
            args.urdf = embodiedment_meta["urdf"]

    assert args.urdf is not None
    logger.info(f"urdf: {args.urdf}")
    checker = PiperMcapChecker(
        input_path=args.input_path,
        output_path=args.output_path,
        urdf=args.urdf,
        task_names=task_names,
        user_names=args.user_names,
        date_prefix=args.date_prefix,
        num_workers=args.num_workers,
        calibration_dict=calibration_dict,
    )
    checker()
