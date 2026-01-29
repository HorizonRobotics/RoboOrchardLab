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
import json
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.behavior import utils
from robo_orchard_lab.dataset.behavior.annotation import Annotation
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)


class BehaviorManipulationPacker(BaseLmdbManipulationDataPacker):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        commit_step: int = 256,
        **kwargs,
    ):
        super().__init__(input_path, output_path, commit_step, **kwargs)


class BehaviorNavigationPacker(BaseLmdbManipulationDataPacker):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        commit_step: int = 256,
        **kwargs,
    ):
        super().__init__(input_path, output_path, commit_step, **kwargs)


class BehaviorPacker:
    """Class of Packing data into lmdb fro behavior1k.

    Return:
        navigation/manipulation lmdb

    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        commit_step: int = 256,
        tasks: list[str] = None,
        num_steps_per_shard: int = None,
        **kwargs,
    ):
        self.tasks = tasks
        self.num_steps_per_shard = num_steps_per_shard


        self.manip_packer = BehaviorManipulationPacker(
            input_path,
            output_path + "_manipulation_lmdb",
            commit_step,
            **kwargs,
        )
        self.nav_packer = BehaviorNavigationPacker(
            input_path,
            output_path + "_navigation_lmdb",
            commit_step,
            **kwargs,
        )

        self.manip_packer._init_lmdbs()
        self.nav_packer._init_lmdbs()

        self.input_path_handler(input_path)

    def _assign_subtask_to_skill(
        self,
        skill: Mapping[str, Any],
        subtasks: Sequence[Mapping[str, Any]],
        min_ratio: float = 0.3,
    ) -> Mapping[str, Any] | None:
        s, e = skill["start"], skill["end"]
        skill_len = e - s
        if skill_len <= 0:
            return None

        best_st = None
        best_ratio = 0.0
        for st in subtasks:
            a, b = st["start"], st["end"]
            overlap = max(0, min(e, b) - max(s, a))
            ratio = overlap / skill_len

            if ratio > best_ratio:
                best_ratio = ratio
                best_st = st

        if best_ratio >= min_ratio:
            return best_st

        return None

    def _build_mobile_traj(
        self,
        robot_pos: np.ndarray,
        robot_ori_sin: np.ndarray,
        robot_ori_cos: np.ndarray,
        yaw_idx: int = -1,
    ) -> np.ndarray:
        """Convert robot position + orientation (sin, cos) to [x, y, yaw].

        Args:
            robot_pos: world position xyz
            robot_ori_sin: orientation sin
            robot_ori_cos: orientation cos
            yaw_idx: which index corresponds to yaw (default: last)

        Returns:
            traj_xy_yaw: [x, y, yaw]
        """
        xy = robot_pos[:, :2]

        yaw = np.arctan2(
            robot_ori_sin[:, yaw_idx],
            robot_ori_cos[:, yaw_idx],
        )
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

        traj_xy_yaw = np.concatenate(
            [xy, yaw[:, None]],
            axis=1,
        )

        return traj_xy_yaw

    def load_skill(
        self,
        anno_file: str,
        task_desc: str,
        pad_amount: int = 0,
    ) -> list[Mapping[str, Any]]:
        """Build skill from annotation.

        Return:
            skill = [start, end, skill_desc, subtask_desc, task_desc]
        """
        anno = Annotation(anno_file)

        # load subtask
        subtasks = []
        for primitive in anno.iter_primitive():
            idx = primitive["primitive_idx"]
            subtask_text = primitive["primitive_text"]
            subtask_id = primitive["primitive_id"]
            start, end = primitive["frame_duration"]
            subtasks.append(
                {
                    "idx": idx,
                    "start": start,
                    "end": end,
                    "subtask_text": subtask_text,
                    "subtask_id": subtask_id

                }

            )
        subtasks.sort(key=lambda x: x["idx"])

        # load skill
        raw_skills = []
        for skill in anno.iter_skill():
            idx = skill["skill_idx"]
            start, end = skill["frame_duration"]
            skill_text = skill["skill_text"]
            skill_id = skill["skill_id"]
            skill_type = skill["skill_type"][0]
            if skill_type in ["uncoordinated", "coordinated"]:
                skill_type = "manipulation"
            elif skill_type in ["navigation"]:
                skill_type = "navigation"
            else:
                skill_type = "navigation"

            raw_skills.append(
                {
                    "idx": idx,
                    "start": start,
                    "end": end,
                    "skill_text": skill_text,
                    "skill_id": skill_id,
                    "skill_type": skill_type,
                    "task_text": task_desc
                }
            )
        raw_skills.sort(key=lambda x: x["idx"])

        # del gap + pad
        pad_skill = []
        for skill in raw_skills:
            start, end = skill["start"], skill["end"]

            if pad_skill:
                prev_skill = pad_skill[-1]

                # two consecutive skills share the same frame_duration
                if start == prev_skill["start"] and end == prev_skill["end"]:
                    start = prev_skill["start"]
                else:
                    min_start = prev_skill["end"]
                    if min_start < end:
                        start = max(min_start, start - pad_amount)

            skill["start"] = start
            skill["end"] = end
            if end - start > 10:
                pad_skill.append(skill)

        # skill to subtask
        for skill in pad_skill:
            st = self._assign_subtask_to_skill(
                skill, subtasks, min_ratio=0.5
            )
            if st is None:
                skill["subtask_id"] = -1
                skill["subtask_text"] = "null"
            else:
                skill["subtask_id"] = st["subtask_id"]
                skill["subtask_text"] = st["subtask_text"]

        return pad_skill

    def load_parquet(self, path: str) -> tuple[
        np.ndarray,  # mobile_traj
        np.ndarray,  # state
        np.ndarray,  # action
        np.ndarray,  # extrinsic
        np.ndarray,  # intrinsic
    ]:
        df = pd.read_parquet(path)

        cam_rel_poses = np.array(
            df["observation.cam_rel_poses"].tolist(), dtype=np.float32
        )
        # task_info = np.array(
        #     df["observation.task_info"].tolist(), dtype=np.float32
        # )
        # timestamp =  np.array(df["timestamp"].tolist(), dtype=np.float32)

        full_obs = np.array(
            df["observation.state"].tolist(), dtype=np.float32
        )

        # joint state
        state = [
            full_obs[:, utils.PROPRIO_QPOS_INDICES["R1Pro"]["torso"]],
            full_obs[:, utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_arm"]],
            full_obs[
                :, utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_gripper"]
            ].sum(axis=-1, keepdims=True),
            full_obs[:, utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_arm"]],
            full_obs[
                :, utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_gripper"]
            ].sum(axis=-1, keepdims=True),
        ]
        state = np.concatenate(state, axis=-1)

        # joint action
        action = np.array(df["action"].tolist(), dtype=np.float32)[:, 3:]

        # robot pose
        robot_pos = full_obs[
            :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["robot_pos"]
        ]
        robot_ori_sin = full_obs[
            :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["robot_ori_sin"]
        ]
        robot_ori_cos = full_obs[
            :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["robot_ori_cos"]
        ]

        # base traj
        mobile_traj = self._build_mobile_traj(
            robot_pos,
            robot_ori_sin,
            robot_ori_cos
        )

        # extrinsic / intrinsic
        cam_names = utils.ROBOT_CAMERA_NAMES["R1Pro"]
        num_cam = len(cam_names)
        num_steps = full_obs.shape[0]

        extrinsic = np.zeros((num_steps, num_cam, 4, 4), dtype=np.float32)
        intrinsic = np.zeros((num_steps, num_cam, 4, 4), dtype=np.float32)

        for cam_idx, cam in enumerate(cam_names):
            cam2base = cam_rel_poses[:, 7 * cam_idx : 7 * cam_idx + 7]
            pos = cam2base[:, :3]
            quat = cam2base[:, 3:]

            # Add camera coordinate system adjustment:
            # 180 degree rotation around X-axis
            rot = Rotation.from_quat(quat).as_matrix()  # (T, 3, 3)
            rot_add = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
            rot_matrix = rot @ rot_add

            extr = np.tile(np.eye(4, dtype=np.float32), (num_steps, 1, 1))
            extr[:, :3, :3] = rot_matrix
            extr[:, :3, 3] = pos
            extrinsic[:, cam_idx] = np.linalg.inv(extr)

            intr = np.tile(np.eye(4, dtype=np.float32), (num_steps, 1, 1))
            intr[:, :3, :3] = utils.CAMERA_INTRINSICS["R1Pro"][cam]
            intrinsic[:, cam_idx] = intr

        return (
            mobile_traj,
            state,
            action,
            extrinsic,
            intrinsic
        )

    def load_frames(
        self,
        cam2path: Mapping[str, str],
    ) -> dict[str, list[np.ndarray]]:
        results = {}
        for cam, path in cam2path.items():
            frames = []
            if "rgb" in cam:
                for frame in utils.decode_video_to_frames_ffmpeg(path):
                    frames.append(frame)
            elif "depth" in cam:
                for frame in utils.decode_depth_to_frames_ffmpeg(path):
                    frames.append(frame)

            results[cam] = frames
        return results

    def input_path_handler(self, input_path: str) -> None:
        self.episodes = []
        for line in open(f"{input_path}/meta/episodes.jsonl"):
            ep_info = json.loads(line.strip())
            ep_index = int(ep_info["episode_index"])
            task_id = f"task-{int(ep_index / 10000):04d}"
            episode_id = f"episode_{ep_index:08d}"

            if self.tasks is not None and task_id not in self.tasks:
                continue

            # parquet: state, action
            parquet_dir = f"{input_path}/data/{task_id}"
            parquet_file = f"{parquet_dir}/{episode_id}.parquet"

            # video: rgbd
            path_prefix = f"{input_path}/videos/{task_id}/observation.images"
            cam2path = {
                "rgb_head": (f"{path_prefix}.rgb.head/{episode_id}.mp4"),
                "rgb_left_wrist": (
                    f"{path_prefix}.rgb.left_wrist/{episode_id}.mp4"
                ),
                "rgb_right_wrist": (
                    f"{path_prefix}.rgb.right_wrist/{episode_id}.mp4"
                ),
                "depth_head": (f"{path_prefix}.depth.head/{episode_id}.mp4"),
                "depth_left_wrist": (
                    f"{path_prefix}.depth.left_wrist/{episode_id}.mp4"
                ),
                "depth_right_wrist": (
                    f"{path_prefix}.depth.right_wrist/{episode_id}.mp4"
                ),
            }

            # annotation: subtask info
            anno_dir = f"{input_path}/annotations/{task_id}"
            anno_file = f"{anno_dir}/{episode_id}.json"

            task_desc = ep_info["tasks"][0]
            assert len(ep_info["tasks"]) == 1

            episode_len = ep_info["length"]

            task_name = utils.TASK_INDICES_TO_NAMES[int(ep_index / 10000)]
            self.episodes.append(
                [
                    task_id,
                    episode_id,
                    parquet_file,
                    anno_file,
                    cam2path,
                    task_desc,
                    task_name,
                    episode_len,
                ]
            )

    def _select_packer(
        self,
        skill_type: str
    ) -> BaseLmdbManipulationDataPacker:
        if skill_type == "navigation":
            return self.nav_packer
        elif skill_type == "manipulation":
            return self.manip_packer
        else:
            raise ValueError(f"Unknown skill_type: {skill_type}")

    def write_index_data(self, key, value, skill_type):
        packer = self._select_packer(skill_type)
        packer.write_index(key, value)

    def write_meta_data(
        self,
        skill_uuid: str,
        skill: dict,
        mobile_traj: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> None:
        skill_type = skill["skill_type"]
        packer = self._select_packer(skill_type)

        packer.meta_pack_file.write(
            f"{skill_uuid}/instruction", skill["task_text"]
        )
        packer.meta_pack_file.write(
            f"{skill_uuid}/subtask_text", skill["subtask_text"]
        )
        packer.meta_pack_file.write(
            f"{skill_uuid}/subtask_id", skill["subtask_id"]
        )
        packer.meta_pack_file.write(
            f"{skill_uuid}/skill_text", skill["skill_text"]
        )
        packer.meta_pack_file.write(
            f"{skill_uuid}/skill_id", skill["skill_id"]
        )

        num_steps = state.shape[0]
        assert state.shape[0] == num_steps, "state length mismatch"
        assert action.shape[0] == num_steps, "action length mismatch"
        assert extrinsic.shape[0] == num_steps, "extrinsic length mismatch"
        assert intrinsic.shape[0] == num_steps, "intrinsic length mismatch"

        temporal_arrays = {
            "observation/robot_state/mobile_traj": mobile_traj,
            "observation/robot_state/joint_position": state,
            "robot_action/joint_position": action,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

        if self.num_steps_per_shard is None:
            for key, value in temporal_arrays.items():
                packer.meta_pack_file.write(
                    f"{skill_uuid}/{key}",
                    value,
                )
        else:
            shard_size = self.num_steps_per_shard
            num_shards = (num_steps + shard_size - 1) // shard_size

            packer.meta_pack_file.write(
                f"{skill_uuid}/num_steps_per_shard", self.num_steps_per_shard
            )

            for shard_idx in range(num_shards):
                start = shard_idx * shard_size
                end = min(start + shard_size, num_steps)

                for key, value in temporal_arrays.items():
                    packer.meta_pack_file.write(
                        f"{skill_uuid}/{shard_idx}/{key}",
                        value[start:end],
                    )

    def write_rgb_data(
        self,
        video_path: str,
        cam: str,
        uuid_prefix: str,
        skills: list[dict],
        episode_keep_indices: np.ndarray,
    ) -> None:
        keep_prefix = np.cumsum(episode_keep_indices, dtype=np.int32)

        skill_idx = 0
        cur = skills[skill_idx]
        frame_idx = 0
        for frame in utils.decode_video_to_frames_ffmpeg(video_path):
            if not episode_keep_indices[frame_idx]:
                frame_idx += 1
                continue

            while frame_idx >= cur["end"]:
                skill_idx += 1
                if skill_idx >= len(skills):
                    return
                cur = skills[skill_idx]

            if frame_idx < cur["start"]:
                frame_idx += 1
                continue

            assert cur["start"] <= frame_idx < cur["end"], (
                f"frame {frame_idx} not in skill {cur['start']}~{cur['end']}"
            )

            base = keep_prefix[cur["start"] - 1] if cur["start"] > 0 else 0
            local_idx = keep_prefix[frame_idx] - base - 1
            uuid = f"{uuid_prefix}{skill_idx}/{cam}/{local_idx}"
            _, fbuf = cv2.imencode(".jpg", frame.astype(np.uint8))

            skill_type = cur["skill_type"]
            packer = self._select_packer(skill_type)
            packer.image_pack_file.write(uuid, fbuf)

            frame_idx += 1

    def write_depth_data(
        self,
        video_path: str,
        cam: str,
        uuid_prefix: str,
        skills: list[dict],
        episode_keep_indices: np.ndarray,
    ) -> None:
        keep_prefix = np.cumsum(episode_keep_indices, dtype=np.int32)

        skill_idx = 0
        cur = skills[skill_idx]
        frame_idx = 0

        for frame in utils.decode_depth_to_frames_ffmpeg(video_path):
            if not episode_keep_indices[frame_idx]:
                frame_idx += 1
                continue

            while frame_idx >= cur["end"]:
                skill_idx += 1
                if skill_idx >= len(skills):
                    return
                cur = skills[skill_idx]

            if frame_idx < cur["start"]:
                frame_idx += 1
                continue

            assert cur["start"] <= frame_idx < cur["end"], (
                f"frame {frame_idx} not in skill {cur['start']}~{cur['end']}"
            )

            frame_scaled = frame * 1000
            _, fbuf = cv2.imencode(
                ".png", frame_scaled.astype(np.uint16)
            )

            base = keep_prefix[cur["start"] - 1] if cur["start"] > 0 else 0
            local_idx = keep_prefix[frame_idx] - base - 1
            uuid = f"{uuid_prefix}{skill_idx}/{cam}/{local_idx}"

            skill_type = cur["skill_type"]
            packer = self._select_packer(skill_type)
            packer.depth_pack_file.write(uuid, fbuf)

            frame_idx += 1

    def process(self, suffix: str = "part") -> None:
        ep_idx = 0
        for ep in self.episodes:
            (
                task_id,
                episode_id,
                parquet_file,
                anno_file,
                cam2path,
                task_desc,
                task_name,
                episode_len,
            ) = ep

            print(int(task_id.split("-")[-1]), episode_id)

            # load skills
            skills = self.load_skill(anno_file, task_desc)

            # load state/action
            (
                mobile_traj,
                state,
                action,
                extrinsic,
                intrinsic,
            ) = self.load_parquet(parquet_file)


            num_steps = state.shape[0]
            episode_keep_indices = np.zeros(num_steps, dtype=bool)
            for i, skill in enumerate(skills):
                skill_uuid = f"{task_id}_{episode_id}_{suffix}{i}"
                start, end = skill["start"], skill["end"]

                (
                    skill_keep,
                    mobile_traj_f,
                    state_f,
                    action_f,
                    extrinsic_f,
                    intrinsic_f,
                ) = utils.compute_episode_keep_indices(
                    mobile_traj=mobile_traj[start:end],
                    state=state[start:end],
                    action=action[start:end],
                    extrinsic=extrinsic[start:end],
                    intrinsic=intrinsic[start:end]
                )

                episode_keep_indices[start:end] |= skill_keep

                self.write_meta_data(
                    skill_uuid=skill_uuid,
                    skill=skill,
                    state=state_f,
                    action=action_f,
                    mobile_traj=mobile_traj_f,
                    extrinsic=extrinsic_f,
                    intrinsic=intrinsic_f,
                )

                # index
                index_data = dict(
                    uuid=skill_uuid,
                    task_name=task_name,
                    num_steps=len(state_f),
                    success=True,
                    simulation=True,
                )
                self.write_index_data(
                    ep_idx,
                    index_data,
                    skill["skill_type"]
                )
                ep_idx += 1

            # not a good idea, but load all frames into memory maybe oom
            for cam, path in cam2path.items():
                uuid_prefix = f"{task_id}_{episode_id}_{suffix}"
                if "rgb" in cam:
                    self.write_rgb_data(
                        video_path=path,
                        cam=cam,
                        uuid_prefix=uuid_prefix,
                        skills=skills,
                        episode_keep_indices=episode_keep_indices
                    )
                elif "depth" in cam:
                    self.write_depth_data(
                        video_path=path,
                        cam=cam,
                        uuid_prefix=uuid_prefix,
                        skills=skills,
                        episode_keep_indices=episode_keep_indices
                    )
        self.manip_packer.close()
        self.nav_packer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("--num_steps_per_shard", type=int)
    args = parser.parse_args()

    print(args)
    bp = BehaviorPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        tasks=args.tasks,
        num_steps_per_shard=args.num_steps_per_shard
    )
    bp.process()
