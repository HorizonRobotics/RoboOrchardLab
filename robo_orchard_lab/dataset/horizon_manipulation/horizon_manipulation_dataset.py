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

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_orchard_lab.dataset.horizon_manipulation.utils import (
    decode_depth,
    decode_img,
    depth_visualize,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class HorizonManipulationLmdbDataset(BaseLmdbManipulationDataset):
    """RoboTwin LMDB Dataset.

    Index structure:

    .. code-block:: text

        {episode_idx}:
            ├── uuid: str
            ├── task_name: str
            ├── user: str
            ├── num_steps: int
            └── simulation: bool

    Meta data structure:

    .. code-block:: text

        {uuid}/meta_data: dict
        {uuid}/camera_names: list(str)
        {uuid}/extrinsic
            └── {cam_name}: np.ndarray[num_steps x 4 x 4] or np.ndarray[4 x 4]
        {uuid}/intrinsic
            ├── {cam_name}: np.ndarray[3 x 3]
        {uuid}/observation/robot_state/cartesian_position
        {uuid}/observation/robot_state/joint_positions
        {uuid}/observation/robot_state/master_joint_positions

    Image storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}

    Depth storage:

    .. code-block:: text

        {uuid}/{cam_name}/{step_idx}
    """

    def __init__(
        self,
        paths,
        transforms=None,
        interval=None,
        load_image=True,
        load_depth=True,
        task_names=None,
        lazy_init=False,
        num_episode=None,
        cam_names=None,
        load_extrinsic=True,
        load_calibration=True,
        load_ee_state=False,
        bgr2rgb=False,
        depth_scale=1000,
        hist_steps=None,
        pred_steps=None,
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            transforms=transforms,
            interval=interval,
            load_image=load_image,
            load_depth=load_depth,
            task_names=task_names,
            lazy_init=lazy_init,
            num_episode=num_episode,
            **kwargs,
        )
        self.cam_names = cam_names
        self.load_extrinsic = load_extrinsic
        self.load_ee_state = load_ee_state
        self.load_calibration = load_calibration
        self.bgr2rgb = bgr2rgb
        self.depth_scale = depth_scale
        self.hist_steps = hist_steps
        self.pred_steps = pred_steps

    def get_instruction(self, lmdb_index, data):
        result = None
        if self.instruction_reader is not None:
            # read frame or episode level instruction
            result = self.instruction_reader.get(
                data["uuid"],
                data["step_index"],
            )
            if result is None:
                # without frame or episode level, read task level instruction
                result = self.instruction_reader.get(data["task_name"])

        if result is None:
            meta = self.meta_lmdbs[lmdb_index][f"{data['uuid']}/meta_data"]
            result = {"instruction": meta.get("instruction"), "subtask": None}

        instruction = result["instruction"]
        if instruction is None or len(instruction) == 0:
            instruction = ""
        elif isinstance(instruction, (list, tuple)):
            idx = np.random.randint(len(instruction))
            instruction = instruction[idx]
        result["text"] = instruction
        return result

    def get_depths(self, lmdb_index, data):
        depths = []
        for cam_name in data["cam_names"]:
            depth_buffer = self.depth_lmdbs[lmdb_index][
                f"{data['uuid']}/{cam_name}/{data['step_index']}"
            ]
            depth = decode_depth(depth_buffer, self.depth_scale)
            depths.append(depth)
        return {"depths": depths}

    def get_images(self, lmdb_index, data):
        images = []
        for cam_name in data["cam_names"]:
            img_buffer = self.img_lmdbs[lmdb_index][
                f"{data['uuid']}/{cam_name}/{data['step_index']}"
            ]
            img = decode_img(img_buffer)
            if self.bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        return {"imgs": images}

    def get_intrinsic(self, lmdb_index, data):
        if "intrinsic" in data:
            intrinsics = data["intrinsic"]
        else:
            intrinsics = self.meta_lmdbs[lmdb_index][
                f"{data['uuid']}/intrinsic"
            ]
        if intrinsics is None:
            camera_info = self.meta_lmdbs[lmdb_index][
                f"{data['uuid']}/camera_info"
            ]
            intrinsics = {}
            for cam_name in camera_info.keys():
                intrinsics[cam_name] = np.array(
                    camera_info[cam_name]["image"]["P"]
                ).reshape(3, 4)
        intrinsic = []
        for cam_name in data["cam_names"]:
            tmp = np.eye(4)
            tmp[:3, :3] = intrinsics[cam_name][:3, :3]
            intrinsic.append(tmp)
        intrinsic = np.stack(intrinsic)
        return {"intrinsic": intrinsic}

    def get_extrinsic(self, lmdb_index, data):
        extrinsics = self.meta_lmdbs[lmdb_index][f"{data['uuid']}/extrinsic"]

        T_world2cam = []  # noqa: N806
        for cam_name in data["cam_names"]:
            # TODO
            if cam_name not in extrinsics:
                assert cam_name == "mid" and "middle" in extrinsics
                cam_name = "middle"
            _ext = extrinsics[cam_name]
            if _ext.ndim == 3:
                _ext = _ext[data["step_index"]]
            T_world2cam.append(_ext)
        T_world2cam = np.stack(T_world2cam)  # noqa: N806
        return {"T_world2cam": T_world2cam}

    def get_calibration(self, lmdb_index, data):
        calibration = self.meta_lmdbs[lmdb_index][
            f"{data['uuid']}/calibration"
        ]
        return {"calibration": calibration}

    def _concat_shards(self, *shards):
        shards = [x for x in shards if x is not None]
        if len(shards) == 0:
            return None
        elif isinstance(shards[0], np.ndarray):
            return np.concatenate(shards, axis=0)
        elif isinstance(shards[0], list):
            results = []
            for x in shards:
                results.extend(x)
            return results

    def _get_meta_with_shard(
        self, lmdb_index, uuid, key, step_index, num_steps_per_shard
    ):
        shard_index = step_index // num_steps_per_shard
        current_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/{shard_index}/{key}"
        ]
        step_index_in_shard = step_index % num_steps_per_shard
        if (
            self.hist_steps is not None
            and step_index_in_shard < self.hist_steps - 1
            and shard_index != 0
        ):
            pre_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index - 1}/{key}"
            ]
        else:
            pre_shard = None

        if (
            self.pred_steps is not None
            and num_steps_per_shard - step_index_in_shard < self.pred_steps
        ):
            next_shard = self.meta_lmdbs[lmdb_index][
                f"{uuid}/{shard_index + 1}/{key}"
            ]
        else:
            next_shard = None
        if pre_shard is not None:
            step_index_in_shard += len(pre_shard)
        data = self._concat_shards(pre_shard, current_shard, next_shard)
        return data, step_index_in_shard

    def get_joint_state(self, lmdb_index, data):
        num_steps_per_shard = data["num_steps_per_shard"]
        uuid = data["uuid"]
        if num_steps_per_shard is None:
            joint_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/joint_positions"
            ]
            master_joint_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/master_joint_positions"
            ]
            step_index_in_shard = data["step_index"]
        else:
            joint_state, step_index_in_shard = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "observation/robot_state/joint_positions",
                data["step_index"],
                num_steps_per_shard,
            )
            master_joint_state = self._get_meta_with_shard(
                lmdb_index,
                uuid,
                "observation/robot_state/master_joint_positions",
                data["step_index"],
                num_steps_per_shard,
            )[0]
        results = {
            "joint_state": np.array(joint_state),
            "step_index_in_shard": step_index_in_shard,
        }
        if master_joint_state is not None:
            results["master_joint_state"] = np.array(master_joint_state)
        return results

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index].get(episode_index)
        )
        uuid = idx_data.uuid
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]
        num_steps_per_shard = self.meta_lmdbs[lmdb_index][
            f"{uuid}/num_steps_per_shard"
        ]
        data = dict(
            uuid=uuid,
            step_index=step_index,
            task_name=idx_data.task_name,
            cam_names=cam_names,
            num_steps_per_shard=num_steps_per_shard,
        )

        data.update(self.get_joint_state(lmdb_index, data))
        if self.load_ee_state:
            ee_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/cartesian_position"
            ]
            data["ee_state"] = np.array(ee_state)

        data.update(self.get_intrinsic(lmdb_index, data))
        if self.load_image:
            data.update(self.get_images(lmdb_index, data))
        if self.load_depth:
            data.update(self.get_depths(lmdb_index, data))
        if self.load_extrinsic:
            data.update(self.get_extrinsic(lmdb_index, data))
        if self.load_calibration:
            data.update(self.get_calibration(lmdb_index, data))

        data.update(self.get_instruction(lmdb_index, data))

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data

    def visualize(
        self,
        episode_index,
        output_path="./vis_data",
        fps=25,
        interval=1,
        **kwargs,
    ):
        from tqdm import tqdm

        end_idx = self.cumsum_steps[episode_index]
        if episode_index != 0:
            start_idx = self.cumsum_steps[episode_index - 1]
        else:
            start_idx = 0
        videoWriter = None  # noqa: N806
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        uuid = self.__getitem__(start_idx)["uuid"]
        file = os.path.join(output_path, f"{uuid.replace('/', '-')}.mp4")

        logger.info(f"episode start_idx: {start_idx}, end_idx: {end_idx}")
        logger.info(f"video save path: {file}")

        for i in tqdm(list(range(start_idx, end_idx, interval))):
            data = self.__getitem__(i)
            if i == start_idx:
                logger.info(
                    f"instruction: {data.get('text')}, "
                    f"subtask: {data.get('subtask')}"
                )

            vis_imgs = self.get_vis_imgs(
                data["imgs"],
                data.get("projection_mat"),
                data.get("hist_robot_state", [None])[-1],
                **kwargs,
            )

            if "depths" in data.keys():
                vis_depths = depth_visualize(data["depths"])
                if len(vis_depths) % 2 == 0:
                    num_imgs = len(vis_depths)
                    vis_depths = np.concatenate(
                        [
                            np.concatenate(
                                vis_depths[: num_imgs // 2], axis=1
                            ),
                            np.concatenate(
                                vis_depths[num_imgs // 2 :], axis=1
                            ),
                        ],
                        axis=0,
                    )
                else:
                    vis_depths = np.concatenate(vis_depths, axis=1)
                vis_imgs = np.concatenate([vis_imgs, vis_depths], axis=0)

            if videoWriter is None:
                videoWriter = cv2.VideoWriter(  # noqa: N806
                    file,
                    fourcc,
                    fps // interval,
                    vis_imgs.shape[:2][::-1],
                )
            videoWriter.write(vis_imgs)
        videoWriter.release()

    @staticmethod
    def get_vis_imgs(
        imgs,
        projection_mat=None,
        robot_state=None,
        ee_indices=(6, 13),
        channel_conversion=False,
    ):
        from scipy.spatial.transform import Rotation

        if isinstance(imgs, torch.Tensor):
            imgs = imgs.cpu().numpy()
        if isinstance(projection_mat, torch.Tensor):
            projection_mat = projection_mat.cpu().numpy()
        if isinstance(robot_state, torch.Tensor):
            robot_state = robot_state.cpu().numpy()

        vis_imgs = []
        for img_index in range(imgs.shape[0]):
            img = imgs[img_index]
            if robot_state is None or projection_mat is None:
                vis_imgs.append(img)
                continue
            for joint_index in range(robot_state.shape[0]):
                rot = Rotation.from_quat(
                    robot_state[joint_index, 4:], scalar_first=True
                ).as_matrix()
                trans = robot_state[joint_index, 1:4]

                if joint_index in ee_indices:
                    axis_length = 0.1
                else:
                    axis_length = 0.03
                points = np.float32(
                    [
                        [axis_length, 0, 0],
                        [0, axis_length, 0],
                        [0, 0, axis_length],
                        [0, 0, 0],
                    ]
                )
                points = points @ rot.T + trans

                pts_2d = points @ projection_mat[img_index, :3, :3].T
                pts_2d = pts_2d + projection_mat[img_index, :3, 3]
                depth = pts_2d[:, 2]
                pts_2d = pts_2d[:, :2] / depth[:, None]

                if depth[3] < 0.02:
                    continue

                pts_2d = pts_2d.astype(np.int32)
                for i in range(3):
                    if depth[i] < 0.02:
                        continue
                    cv2.circle(
                        img, (pts_2d[i, 0], pts_2d[i, 1]), 6, (0, 0, 255), -1
                    )
                    if i == 3:
                        continue
                    color = [0, 0, 0]
                    color[i] = 255
                    cv2.line(
                        img,
                        (pts_2d[i, 0], pts_2d[i, 1]),
                        (pts_2d[3, 0], pts_2d[3, 1]),
                        tuple(color),
                        3,
                    )
            vis_imgs.append(img)

        if len(vis_imgs) % 2 == 0:
            num_imgs = len(vis_imgs)
            vis_imgs = np.concatenate(
                [
                    np.concatenate(vis_imgs[: num_imgs // 2], axis=1),
                    np.concatenate(vis_imgs[num_imgs // 2 :], axis=1),
                ],
                axis=0,
            )
        else:
            vis_imgs = np.concatenate(vis_imgs, axis=1)
        vis_imgs = np.uint8(vis_imgs)
        if channel_conversion:
            vis_imgs = vis_imgs[..., ::-1]
        return vis_imgs


class RH20TManipulationDataset(HorizonManipulationLmdbDataset):
    def __init__(self, num_views=-1, time_threshold=200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self.time_threshold = time_threshold  # ms
        self.load_extrinsic = True

    def _check_valid(self, index_data):
        if index_data.num_steps > 900:
            return False
        return super()._check_valid(index_data)

    def sample_camera(
        self,
        camera_names,
        meta,
        step_index,
        intrinsics,
        extrinsics,
    ):
        valid = np.array(
            [
                (name in intrinsics) and (name in extrinsics)
                for name in camera_names
            ]
        )
        if self.time_threshold is not None:
            valid_thred = np.array(
                [
                    abs(meta["image_time_shift"][name][step_index])
                    < self.time_threshold
                ]
                for name in camera_names
            )
            valid = np.logical_and(valid_thred, valid)
        wrist = np.array([name in meta["wrist_cam"] for name in camera_names])

        valid_wrist_id = np.where(np.logical_and(valid, wrist))[0]
        valid_env_id = np.where(np.logical_and(valid, np.logical_not(wrist)))[
            0
        ]

        sample_wrist, sample_env = 1, self.num_views - 1

        if len(valid_wrist_id) + len(valid_env_id) < self.num_views:
            return None

        if len(valid_wrist_id) == 0:
            sample_env += 1
            sample_wrist_id = np.array([])

        if len(valid_env_id) == 0:
            sample_wrist = self.num_views
            sample_env_id = np.array([])

        if len(valid_wrist_id) > 0:
            sample_wrist_id = np.random.choice(
                valid_wrist_id,
                sample_wrist,
                replace=len(valid_wrist_id) < sample_wrist,
            )
        if len(valid_env_id) > 0:
            sample_env_id = np.random.choice(
                valid_env_id,
                sample_env,
                replace=len(valid_env_id) < sample_env,
            )
        sample_id = np.concatenate([sample_env_id, sample_wrist_id])
        camera_names = [camera_names[int(i)] for i in sample_id]
        if len(camera_names) < self.num_views:
            return None
        return camera_names

    def get_depths(self, lmdb_index, data):
        depths = []
        for cam_name in data["cam_names"]:
            depth_buffer = self.depth_lmdbs[lmdb_index][
                f"{data['uuid']}/{cam_name}/{data['step_index']}"
            ]
            if cam_name[0].isalpha():
                depth_scale = 4000
            else:
                depth_scale = 1000
            depth = (
                cv2.imdecode(
                    np.frombuffer(depth_buffer, np.uint8), cv2.IMREAD_UNCHANGED
                )
                / depth_scale
            )
            depths.append(depth)
        return {"depths": depths}

    def get_extrinsic(self, lmdb_index, data):
        step_index = data["step_index"]
        ee_state = data["ee_state"][[0, step_index]]
        trans_mat = np.eye(4)[None].repeat(2, 0)
        trans_mat[:, :3, :3] = Rotation.from_quat(
            ee_state[:, 3:], scalar_first=True
        ).as_matrix()
        trans_mat[:, :3, 3] = ee_state[:, :3]
        trans_mat = trans_mat[0] @ np.linalg.inv(trans_mat[1])

        extrinsics = data.pop("extrinsic")
        wrist_cam = data.pop("wrist_cam")
        T_world2cam = []  # noqa: N806
        for cam_name in data["cam_names"]:
            _ext = extrinsics[cam_name]
            if cam_name in wrist_cam:
                _ext = _ext @ trans_mat
            T_world2cam.append(_ext)
        T_world2cam = np.stack(T_world2cam)  # noqa: N806
        return {"T_world2cam": T_world2cam}

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index].get(episode_index)
        )
        uuid = idx_data.uuid
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        intrinsics = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]
        extrinsics = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]
        meta = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
        if self.num_views > 0:
            cam_names = self.sample_camera(
                cam_names, meta, step_index, intrinsics, extrinsics
            )
            if cam_names is None:
                index = np.random.randint(self.__len__())
                return self[index]

        joint_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/joint_positions"
        ]
        ee_state = self.meta_lmdbs[lmdb_index][
            f"{uuid}/observation/robot_state/cartesian_position"
        ]
        data = dict(
            uuid=uuid,
            step_index=step_index,
            cam_names=cam_names,
            task_name=idx_data.task_name,
            intrinsic=intrinsics,
            extrinsic=extrinsics,
            wrist_cam=idx_data.wrist_cam,
            joint_state=np.array(joint_state),
            ee_state=np.array(ee_state),
        )

        data.update(self.get_intrinsic(lmdb_index, data))
        data.update(self.get_extrinsic(lmdb_index, data))
        if self.load_image:
            data.update(self.get_images(lmdb_index, data))
        if self.load_depth:
            data.update(self.get_depths(lmdb_index, data))

        data.update(self.get_instruction(lmdb_index, data))

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)
        return data
