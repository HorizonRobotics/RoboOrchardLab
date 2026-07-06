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

import cv2
import numpy as np
import torch

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseIndexData,
    BaseLmdbManipulationDataset,
)

logger = logging.getLogger(__name__)


class ABC130kLmdbDataset(BaseLmdbManipulationDataset):
    """ABC-130K LMDB dataset.

    This dataset follows the same LMDB organization as horizon manipulation:
    ``index / meta / image / depth``. Intrinsics and extrinsics have been
    packed with a per-episode correction pass (see
    ``abc130k_export_lmdb_packer.py``):

    * ``intrinsic_corrected``: K reconciled with the saved image resolution
      (some publisher K's belonged to a different D405 streaming mode).
    * ``extrinsic_corrected``: per-step ``T_world2cam`` computed via URDF FK
      so wrist cameras track the arm; the field is optional and we fall back
      to the static ``extrinsic`` (zero-joint reference) when it's missing.

    Instructions are baked into the pack at ``{uuid}/instructions`` (and
    also mirrored inside ``{uuid}/meta_data``) by the packer, so the
    dataset does not accept an ``instructions`` override — it simply reads
    what's there and falls back to ``task_name`` if both slots are empty.

    ``T_base2world`` / ``T_base2ego`` are *not* dataset attributes — inject
    them via ``AddItems`` in the transform pipeline (see
    ``config_abc130k_dataset.py``), matching horizon's config conventions.
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
        cam_names=None,
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
            **kwargs,
        )
        self.cam_names = cam_names

    def _decode_image(self, encoded):
        if encoded is None:
            return None
        if isinstance(encoded, bytes):
            encoded = np.frombuffer(encoded, np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    def _decode_depth(self, encoded):
        if encoded is None:
            return None
        if isinstance(encoded, bytes):
            encoded = np.frombuffer(encoded, np.uint8)
        depth = cv2.imdecode(
            encoded, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
        )
        return depth.astype(np.float32) / 1000.0

    def get_images(self, lmdb_index, data):
        images = []
        for cam_name in data["cam_names"]:
            image = self.img_lmdbs[lmdb_index][
                f"{data['uuid']}/{cam_name}/{data['step_index']}"
            ]
            images.append(self._decode_image(image))
        return {"imgs": np.stack(images)}

    def get_depths(self, lmdb_index, data):
        depths = []
        for cam_name in data["cam_names"]:
            depth = self.depth_lmdbs[lmdb_index][
                f"{data['uuid']}/{cam_name}/{data['step_index']}"
            ]
            depths.append(self._decode_depth(depth))
        return {"depths": np.stack(depths)}

    def get_intrinsic(self, lmdb_index, data):
        """Read `intrinsic_corrected` (K reconciled with image size).

        Falls back to the legacy ``intrinsic`` field if a pack predates the
        correction (should be re-packed for consistency, but we don't crash).
        """
        uuid = data["uuid"]
        intrinsics = self.meta_lmdbs[lmdb_index].get(
            f"{uuid}/intrinsic_corrected"
        )
        if intrinsics is None:
            intrinsics = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]
        intrinsic = []
        for cam_name in data["cam_names"]:
            tmp = np.eye(4, dtype=np.float64)
            if isinstance(intrinsics, dict) and cam_name in intrinsics:
                k = np.asarray(intrinsics[cam_name], dtype=np.float64)
                if k.shape == (3, 3):
                    tmp[:3, :3] = k
                elif k.shape[0] >= 3 and k.shape[1] >= 3:
                    tmp[:3, :3] = k[:3, :3]
            intrinsic.append(tmp)
        return {"intrinsic": np.stack(intrinsic)}

    def get_extrinsic(self, lmdb_index, data):
        """Read per-step `extrinsic_corrected`, falling back to zero-joint.

        The packer runs URDF FK to bake ``[num_steps, 4, 4]`` per camera; old
        packs only ship the static ``[4, 4]`` reference under ``extrinsic``,
        which we broadcast at the current step.
        """
        uuid = data["uuid"]
        step_index = data["step_index"]
        extrinsics = self.meta_lmdbs[lmdb_index].get(
            f"{uuid}/extrinsic_corrected"
        )
        if extrinsics is None:
            extrinsics = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]

        T_world2cam = []  # noqa: N806
        for cam_name in data["cam_names"]:
            tmp = np.eye(4, dtype=np.float64)
            source = (
                extrinsics.get(cam_name)
                if isinstance(extrinsics, dict)
                else None
            )
            if source is not None:
                source = np.asarray(source, dtype=np.float64)
                if source.ndim == 3:
                    tmp[:3] = source[step_index][:3]
                elif source.ndim == 2:
                    tmp[:3] = source[:3]
            T_world2cam.append(tmp)
        return {"T_world2cam": np.stack(T_world2cam)}

    def get_joint_state(self, lmdb_index, data):
        """Load joint_state (+ master_joint_state) with sharded-window support.

        Mirrors horizon's ``get_joint_state``: reads either flat or sharded
        meta layout and always exposes ``step_index_in_shard`` so state
        samplers can index the window regardless of layout.
        """
        uuid = data["uuid"]
        step_index = data["step_index"]
        num_steps_per_shard = self.meta_lmdbs[lmdb_index].get(
            f"{uuid}/num_steps_per_shard"
        )
        if num_steps_per_shard is None:
            joint_state = self.meta_lmdbs[lmdb_index][
                f"{uuid}/observation/robot_state/joint_positions"
            ]
            master_joint_state = self.meta_lmdbs[lmdb_index].get(
                f"{uuid}/observation/robot_state/master_joint_positions"
            )
            step_index_in_shard = step_index
        else:
            joint_state = self._get_meta(
                lmdb_index,
                uuid,
                "observation/robot_state/joint_positions",
                step_index,
                num_steps_per_shard,
            )
            master_joint_state = self._get_meta(
                lmdb_index,
                uuid,
                "observation/robot_state/master_joint_positions",
                step_index,
                num_steps_per_shard,
            )
            step_index_in_shard = self._get_step_index_in_shard(
                step_index,
                num_steps_per_shard,
            )
        results = {
            "joint_state": np.asarray(joint_state),
            "step_index_in_shard": step_index_in_shard,
            "episode_step_index": step_index,
        }
        # Commanded action (0/1 on gripper columns) vs `joint_state`'s
        # post-contact finger distance. horizon's SimpleStateSampling uses
        # this to swap gripper columns into pred so BC targets can actually
        # close on new objects.
        if master_joint_state is not None:
            results["master_joint_state"] = np.asarray(master_joint_state)
        return results

    def get_instruction(self, lmdb_index, data):
        """Resolve the text prompt for this sample from the pack.

        Precedence:
            1. episode-level ``{uuid}/instructions`` written by the packer
               (from the MCAP ``/instruction`` topic);
            2. ``{uuid}/meta_data['instruction']`` as a legacy fallback;
            3. ``task_name`` so we never emit an empty prompt.

        A list is randomly sampled once per __getitem__ call, matching
        horizon's behavior — packers may pack a single string or a list of
        paraphrases and the dataset handles both.
        """
        uuid = data["uuid"]
        task_name = data["task_name"]

        instructions = self.meta_lmdbs[lmdb_index][f"{uuid}/instructions"]
        if instructions is None:
            meta_data = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
            if isinstance(meta_data, dict):
                instructions = meta_data.get("instruction")

        if isinstance(instructions, (list, tuple)) and len(instructions) > 0:
            text = instructions[np.random.randint(len(instructions))]
        elif isinstance(instructions, str):
            text = instructions
        else:
            text = ""
        if not text:
            text = task_name
        return {"text": text}

    def __getitem__(self, index):
        lmdb_index, episode_index, step_index = self._get_indices(index)

        idx_data = BaseIndexData.model_validate(
            self.idx_lmdbs[lmdb_index][episode_index]
        )
        uuid = idx_data.uuid
        task_name = idx_data.task_name
        if self.cam_names is not None:
            cam_names = self.cam_names
        else:
            cam_names = self.meta_lmdbs[lmdb_index][f"{uuid}/camera_names"]

        data = dict(
            uuid=uuid,
            task_name=task_name,
            step_index=step_index,
            cam_names=cam_names,
        )

        data.update(self.get_joint_state(lmdb_index, data))
        data.update(self.get_intrinsic(lmdb_index, data))
        data.update(self.get_extrinsic(lmdb_index, data))
        if self.load_image:
            data.update(self.get_images(lmdb_index, data))
        if self.load_depth:
            data.update(self.get_depths(lmdb_index, data))
        data.update(self.get_instruction(lmdb_index, data))

        # ABC130k's `cartesian_position` uses an undocumented tool offset that
        # does not match the official URDF, so ee pose is derived downstream
        # via DualArmKinematics on `joint_state` — no `ee_state` here.

        for transform in self.transforms:
            if transform is None:
                continue
            data = transform(data)

        # Keep compatibility with visualizers that expect a `depths` key.
        # ABC130k is RGB-only; provide a zero depth placeholder at runtime.
        if "depths" not in data and "imgs" in data:
            imgs = data["imgs"]
            if isinstance(imgs, torch.Tensor):
                data["depths"] = torch.zeros(
                    (imgs.shape[0], imgs.shape[1], imgs.shape[2]),
                    dtype=torch.float32,
                    device=imgs.device,
                )
            elif isinstance(imgs, np.ndarray):
                data["depths"] = np.zeros(
                    (imgs.shape[0], imgs.shape[1], imgs.shape[2]),
                    dtype=np.float32,
                )
        return data
