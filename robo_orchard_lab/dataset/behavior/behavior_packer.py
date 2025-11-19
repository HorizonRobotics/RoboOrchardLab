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
import concurrent
import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import as_completed
from typing import List

import cv2
import numpy as np
import pandas as pd

from robo_orchard_lab.dataset.behavior import utils
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BehaviorPacker(BaseLmdbManipulationDataPacker):
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
        commit_step=2048,
        sample_rate=3,
        tasks=None,
        multi_proc=True,
        **kwargs,
    ):
        super().__init__(input_path, output_path, commit_step, **kwargs)
        self.sample_rate = sample_rate
        self.tasks = tasks
        self.input_path_handler(input_path)
        self.multi_proc = multi_proc

    def load_parquet(self, path: str) -> dict:
        #################################################################
        # index episode_index task_index timestamp
        # observation.state action
        # observation.cam_rel_poses
        # observation.task_info
        #################################################################

        df = pd.read_parquet(path)
        ret = {
            "state": np.array(
                df["observation.state"].tolist(), dtype=np.float32
            ),
            "action": np.array(df["action"].tolist(), dtype=np.float32),
            "cam_rel_poses": np.array(
                df["observation.cam_rel_poses"].tolist(), dtype=np.float32
            ),
            "task_info": np.array(
                df["observation.task_info"].tolist(), dtype=np.float32
            ),
            "timestamp": np.array(df["timestamp"].tolist(), dtype=np.float32),
        }
        return ret

    def load_frames(self, cam2path) -> dict:
        results = {}
        if self.multi_proc:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=2
            ) as executor:
                future2cam = {
                    executor.submit(
                        utils.decode_video, path, self.sample_rate
                    ): cam
                    for cam, path in cam2path.items()
                }
                for future in as_completed(future2cam):
                    cam = future2cam[future]
                    try:
                        frames = future.result()
                        results[cam] = frames
                    except Exception as e:
                        print(
                            f"[Error] Failed to extract frames for {cam}: {e}"
                        )
                        results[cam] = None
        else:
            for cam, path in cam2path.items():
                frames = utils.decode_video(path, self.sample_rate)
                results[cam] = frames
        return results

    def input_path_handler(self, input_path: str) -> List:
        self.episodes = []
        for line in open(f"{input_path}/meta/episodes.jsonl"):
            ep_info = json.loads(line.strip())
            ep_index = int(ep_info["episode_index"])
            task_id = f"task-{int(ep_index / 10000):04d}"
            episode_id = f"episode_{ep_index:08d}"

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
            # for f in [parquet_file,
            #   #anno_file,
            #  rgb_head_video, rgb_left_video, rgb_right_video,
            #  depth_head_video, depth_left_video, depth_right_video
            # ]:
            #    assert os.path.exists(f), f'{f} does not exists.'

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

    def _pack(self) -> None:
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

            if self.tasks is not None and task_id not in self.tasks:
                continue

            print(int(task_id.split("-")[-1]), episode_id)

            uuid = f"{task_id}_{episode_id}"
            if ep_idx % 1000:
                logger.info(f"start process: {ep_idx}/{len(self.episodes)}")

            self.meta_pack_file.write(
                f"{uuid}/camera_names", utils.ROBOT_CAMERA_NAMES["R1Pro"]
            )
            self.meta_pack_file.write(f"{uuid}/instruction", task_desc)

            # load state/action
            data = self.load_parquet(parquet_file)

            state = data["state"]
            joint_pos = data["state"][
                :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["joint_qpos"]
            ]
            eef_pos = np.hstack(
                [
                    data["state"][
                        :,
                        utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_pos"],
                    ],
                    data["state"][
                        :,
                        utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_quat"],
                    ],
                    data["state"][
                        :,
                        utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_pos"],
                    ],
                    data["state"][
                        :,
                        utils.PROPRIOCEPTION_INDICES["R1Pro"][
                            "eef_right_quat"
                        ],
                    ],
                ]
            )
            base_qvel = data["state"][
                :, utils.PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]
            ]
            action = data["action"]

            # fps sample: 30 --> 10
            state = state[:: self.sample_rate, :]
            joint_pos = joint_pos[:: self.sample_rate, :]
            eef_pos = eef_pos[:: self.sample_rate, :]
            action = action[:: self.sample_rate, :]
            base_qvel = base_qvel[:: self.sample_rate, :]

            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/ori_state", state
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions", joint_pos
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/cartesian_position", eef_pos
            )
            self.meta_pack_file.write(f"{uuid}/action", action)
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/base_qvel", base_qvel
            )

            # extrinsic/intrinsic
            extrinsic = {}
            for idx, cam in enumerate(utils.ROBOT_CAMERA_NAMES["R1Pro"]):
                cam2base = data["cam_rel_poses"][
                    :: self.sample_rate, 7 * idx : 7 * idx + 7
                ]
                b = cam2base.shape[0]
                pos = cam2base[:, :3]  # (b, 3)
                quat = cam2base[:, 3:]  # (b, 4)
                rot = utils.quat2mat(quat)  # (b, 3, 3)

                # Add camera coordinate system adjustment:
                # 180 degree rotation around X-axis
                rot_add = utils.euler2mat([np.pi, 0, 0])  # (3, 3)
                rot_matrix = np.matmul(rot, rot_add)  # (b, 3, 3)

                extr = np.tile(
                    np.eye(4, dtype=float), (b, 1, 1)
                )  # shape (b, 4, 4)
                extr[:, :3, :3] = rot_matrix
                extr[:, :3, 3] = pos
                extrinsic[cam] = extr

            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsic)
            self.meta_pack_file.write(
                f"{uuid}/intrinsic", utils.CAMERA_INTRINSICS["R1Pro"]
            )

            # load rgb/depth frame
            # timestamps = data['timestamp'][::self.sample_rate]

            start = time.time()
            cam2frames = self.load_frames(cam2path)
            end = time.time()
            print(f"load frame: {end - start:.2f}s")
            # for cam, frames in cam2frames.items():
            #    print(len(frames), len(joint_pos))
            #    #cam2frames[cam] = frames[::self.sample_rate, :]

            global_min = np.inf
            global_max = -np.inf
            for cam, frames in cam2frames.items():
                for i, frame in enumerate(frames):
                    if "rgb" in cam:
                        _, fbuf = cv2.imencode(".jpg", frame.astype(np.uint8))
                        self.image_pack_file.write(f"{uuid}/{cam}/{i}", fbuf)
                    elif "depth" in cam:
                        # frame = dequantize_depth(frame)
                        frame_scaled = frame * 1000
                        _, fbuf = cv2.imencode(
                            ".png", frame_scaled.astype(np.uint16)
                        )
                        self.depth_pack_file.write(f"{uuid}/{cam}/{i}", fbuf)

                        val_min = frame.min()
                        val_max = frame.max()
                        if val_min < global_min:
                            global_min = val_min
                        if val_max > global_max:
                            global_max = val_max
                    else:
                        raise ValueError(f"{cam} not found")

            assert len(state) == len(joint_pos), (
                f"shape mismatch:"
                f"state {len(state)} vs joint_pos {len(joint_pos)}"
            )
            assert len(joint_pos) == len(eef_pos), (
                f"shape mismatch:joint: {len(joint_pos)} vs eef {len(eef_pos)}"
            )
            assert len(joint_pos) == len(action), (
                f"shape: joint:{len(joint_pos)} vs action: {len(action)}"
            )
            assert len(joint_pos) == len(base_qvel), (
                f"shape mismatch:"
                f"joint {len(joint_pos)} vs base_qvel{len(base_qvel)}"
            )
            assert len(joint_pos) == len(extrinsic["head"]), (
                f"shape mismatch:"
                f"joint {len(joint_pos)} vs extrinsic {len(extrinsic['head'])}"
            )
            assert len(joint_pos) == len(cam2frames["rgb_head"]), (
                f"shape mismatch:"
                f"joint {len(joint_pos)} vs rgb {len(cam2frames['rgb_head'])}"
            )
            assert len(joint_pos) == len(cam2frames["depth_head"]), (
                f"shape mismatch:"
                f"joint {len(joint_pos)} vs "
                f"depth {len(cam2frames['depth_head'])}"
            )

            # index
            index_data = dict(
                uuid=uuid,
                task_name=task_name,
                num_steps=len(joint_pos),
                success=True,
                simulation=True,
            )
            self.write_index(ep_idx, index_data)
            ep_idx += 1

        self.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("--multi_proc", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=3)
    args = parser.parse_args()

    print(args)
    bp = BehaviorPacker(
        input_path=args.input_path,
        output_path=args.output_path,
        tasks=args.tasks,
        sample_rate=args.sample_rate,
    )
    bp()
