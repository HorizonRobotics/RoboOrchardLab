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

import numpy as np

from robo_orchard_lab.dataset.horizon_manipulation.horizon_manipulation_dataset import (  # noqa: E501
    HorizonManipulationLmdbDataset,
)

logger = logging.getLogger(__name__)


class ABC130kLmdbDataset(HorizonManipulationLmdbDataset):
    """ABC-130K LMDB dataset.

    Same LMDB layout as horizon manipulation (``index / meta / image``); the
    only ABC130K-specific bits are:

    * intrinsics were reconciled against the saved image resolution by the
      packer and stored under ``{uuid}/intrinsic_corrected`` (some publisher
      K's belonged to a different D405 streaming mode);
    * the source dataset ships static extrinsics that don't move with the
      wrist cameras, so the packer runs URDF FK per step and stores the
      result under ``{uuid}/extrinsic_corrected``;
    * text prompts are baked in ``{uuid}/instructions`` (from the MCAP
      ``/instruction`` topic);
    * the source dataset has no depth stream. The packer sets
      ``{uuid}/has_depth = False`` and skips the depth LMDB entirely, so
      ``get_depths`` fabricates zero depths matching the RGB shape rather
      than crashing. It's a short-term hack — every depth-consuming module
      (BatchDepthProbGTGenerator, DepthFusionSpatialEnhancer,
      HolobrainDataFeature) already handles zero depth as "invalid, weight
      0", so the pipeline behaves as if depth loss is off for ABC130K
      samples without having to fork model configs. A re-pack that
      actually decodes the D405 depth stream will remove the need for this
      override.

    We inherit horizon's ``__getitem__`` (which already handles shard-aware
    joint state, cam name resolution, and transform composition) and only
    override the readers that need to route to the ``_corrected`` slots,
    read from the ABC130K instruction slot, or fabricate depth.

    ``load_calibration``/``load_ee_state`` default to False because ABC130K
    has no packed calibration and its ``cartesian_position`` uses an
    undocumented tool offset (ee pose is derived downstream via
    ``MultiArmKinematics`` on ``joint_state``).
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
        load_extrinsic=True,
        load_calibration=False,
        load_ee_state=False,
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
            cam_names=cam_names,
            load_extrinsic=load_extrinsic,
            load_calibration=load_calibration,
            load_ee_state=load_ee_state,
            **kwargs,
        )


    def get_intrinsic(self, lmdb_index, data):
        """Prefer ``intrinsic_corrected`` (K reconciled with image size).

        Falls back to the legacy ``intrinsic`` field if a pack predates the
        correction (should be re-packed for consistency, but we don't crash).
        Same 3x3 -> 4x4 padding as horizon.
        """
        uuid = data["uuid"]
        intrinsics = self.meta_lmdbs[lmdb_index].get(
            f"{uuid}/intrinsic_corrected"
        )
        if intrinsics is None:
            intrinsics = self.meta_lmdbs[lmdb_index][f"{uuid}/intrinsic"]
        intrinsic = []
        for cam_name in data["cam_names"]:
            tmp = np.eye(4)
            tmp[:3, :3] = np.asarray(intrinsics[cam_name])[:3, :3]
            intrinsic.append(tmp)
        return {"intrinsic": np.stack(intrinsic)}

    def get_extrinsic(self, lmdb_index, data):
        """Prefer per-step ``extrinsic_corrected``, else broadcast the static.

        The packer runs URDF FK to bake ``[num_steps, 4, 4]`` per camera; old
        packs only ship the static ``[4, 4]`` reference under ``extrinsic``,
        which we broadcast at the current step (matching horizon's ndim=2
        fallback branch).
        """
        uuid = data["uuid"]
        extrinsics = self.meta_lmdbs[lmdb_index].get(
            f"{uuid}/extrinsic_corrected"
        )
        if extrinsics is None:
            extrinsics = self.meta_lmdbs[lmdb_index][f"{uuid}/extrinsic"]

        T_world2cam = []  # noqa: N806
        for cam_name in data["cam_names"]:
            _ext = np.asarray(extrinsics[cam_name])
            if _ext.ndim == 3:
                _ext = _ext[data["step_index"]]
            T_world2cam.append(_ext)
        return {"T_world2cam": np.stack(T_world2cam)}

    def get_instruction(self, lmdb_index, data):
        """Resolve the text prompt, honoring ``instruction_reader`` first.

        Precedence:
            1. ``self.instruction_reader`` (frame/episode/task level, same as
               horizon) — lets us paraphrase-inject without re-packing;
            2. ``{uuid}/instructions`` written by the packer from MCAP's
               ``/instruction`` topic;
            3. ``{uuid}/meta_data['instruction']`` as a legacy fallback;
            4. ``task_name`` so we never emit an empty prompt.

        A list is randomly sampled once per __getitem__ call. ``subtask`` is
        forwarded through when the reader provides one, matching horizon's
        return shape.
        """
        uuid = data["uuid"]
        result = None
        if self.instruction_reader is not None:
            result = self.instruction_reader.get(uuid, data["step_index"])
            if result is None:
                result = self.instruction_reader.get(data["task_name"])

        if result is None:
            instructions = self.meta_lmdbs[lmdb_index].get(
                f"{uuid}/instructions"
            )
            if instructions is None:
                meta = self.meta_lmdbs[lmdb_index][f"{uuid}/meta_data"]
                if isinstance(meta, dict):
                    instructions = meta.get("instruction")
            result = {"instruction": instructions, "subtask": None}

        instruction = result.get("instruction")
        if isinstance(instruction, (list, tuple)) and len(instruction) > 0:
            instruction = instruction[np.random.randint(len(instruction))]
        elif not isinstance(instruction, str):
            instruction = ""
        if not instruction:
            instruction = data["task_name"]
        result["text"] = instruction

        subtask = result.get("subtask")
        if isinstance(subtask, (list, tuple)) and len(subtask) > 0:
            result["subtask"] = subtask[np.random.randint(len(subtask))]
        return result
