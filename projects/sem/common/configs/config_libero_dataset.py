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

from dataset_factory import processor_register, train_dataset_register

data_paths = dict(
    libero_goal="./data/libero/lmdb_goal_abs",
    libero_object="./data/libero/lmdb_object_abs",
    libero_spatial="./data/libero/lmdb_spatial_abs",
    libero_10="./data/libero/lmdb_10_abs",
)

cam_names = ["eye_in_hand", "agentview"]


def build_transforms(config, mode):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        ConvertDataType,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        MoveEgoToCam,
        Resize,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.dataset.libero.transforms import (
        SimpleStateSampling,
        TransformRobotState,
    )

    t_base2ego = np.eye(4).tolist()
    t_base2world = np.eye(4).tolist()
    joint_mask = (False,)
    joint_scale_shift = ((0.5, 0.5),)
    joint_relative_pos = (0,)
    loss_weights = np.array([[1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]])
    # loss_weights = np.ones([1, 8])
    temporal_weights = (
        np.concatenate(
            [
                np.ones(config["pred_steps"] // 4),
                np.linspace(
                    1.0,
                    0.1,
                    config["pred_steps"] // 2 - config["pred_steps"] // 4,
                ),
                np.zeros(config["pred_steps"] // 2),
            ]
        )[:, None, None]
        * 4
    )
    loss_weights = loss_weights * temporal_weights
    loss_weights = loss_weights.tolist()

    add_data_relative_items = dict(
        type=AddItems,
        T_base2ego=t_base2ego,
        T_base2world=t_base2world,
        joint_mask=joint_mask,
        joint_relative_pos=joint_relative_pos,
        joint_scale_shift=joint_scale_shift,
    )
    if mode == "training":
        add_data_relative_items.update(
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
        )

    add_non_array_items = dict(
        type=AddItems, noise_type="local_joint_local_pose", to_numpy=False
    )

    transform_robot_state = dict(type=TransformRobotState)
    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])

    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam, cam_idx="agentview")
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
            hist_robot_state="float32",
            pred_robot_state="float32",
            joint_scale_shift="float32",
        ),
    )

    if mode == "training":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "joint_relative_pos",
                "text",
                "uuid",
                "subtask",
                "joint_mask",
                "noise_type",
                "state_loss_weights",
                "fk_loss_weight",
                "pred_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            add_non_array_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            transform_robot_state,
            convert_dtype,
            item_selection,
        ]
    elif mode == "validation":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "joint_relative_pos",
                "kinematics",
                "text",
                "uuid",
                "subtask",
                "joint_mask",
                "noise_type",
            ],
        )
        transforms = [
            add_data_relative_items,
            add_non_array_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            transform_robot_state,
            convert_dtype,
            item_selection,
        ]
    elif mode == "deploy":
        item_selection = dict(
            type=ItemSelection,
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "joint_relative_pos",
                "kinematics",
                "text",
                "uuid",
                "subtask",
                "joint_mask",
                "noise_type",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            add_non_array_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            convert_dtype,
            transform_robot_state,
            item_selection,
            unsqueeze_batch,
        ]

    return transforms


@train_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    import uuid

    from robo_orchard_lab.dataset.libero.libero_lmdb_dataset import (
        LiberoLmdbDataset,
    )

    datasets = []
    for dataset_name, data_path in data_paths.items():
        if "libero" not in dataset_names and dataset_name not in dataset_names:
            continue
        transforms = build_transforms(
            config,
            mode,
        )
        dataset = LiberoLmdbDataset(
            paths=data_path,
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            cam_names=cam_names,
            dataset_name=dataset_name,
            flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "libero").hex[:4], 16),
        )
        datasets.append(dataset)
    return datasets


@processor_register()
def build_processors(config, dataset_names):
    from robo_orchard_lab.models.sem_modules.processor import (
        SEMProcessor,
        SEMProcessorCfg,
    )

    processors = {}
    if "libero" in dataset_names:
        transforms = build_transforms(
            config,
            "deploy",
        )
        processor = SEMProcessor(
            SEMProcessorCfg(
                load_image=True,
                load_depth=config["with_depth"],
                valid_action_step=None,
                transforms=transforms,
                cam_names=cam_names,
            )
        )
        processors["libero"] = processor
    return processors
