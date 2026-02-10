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

from dataset_factory import train_dataset_register

data_paths = [
    "./data/egodex/lmdb/part1",
    "./data/egodex/lmdb/part2",
    "./data/egodex/lmdb/part3",
    "./data/egodex/lmdb/part4",
    "./data/egodex/lmdb/part5",
    "./data/egodex/lmdb/extra",
    "./data/egodex/lmdb/test",
]


def build_transforms(config, mode):
    import numpy as np
    import torch

    from robo_orchard_lab.dataset.egodex.transforms import (
        HandTF2Gripper,
        SimpleStateSampling,
        UpSampleRobotState,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        Resize,
        ToTensor,
    )

    hand_to_gripper = HandTF2Gripper()
    scale_shift = [[0.06, 0.06], [0.06, 0.06]]
    ee_state_loss_weights = [0.1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    loss_weights = np.array([[ee_state_loss_weights]])
    egodex_loss_weight = 0.5
    state_loss_weights = loss_weights * egodex_loss_weight
    joint_mask = np.array([False, False])

    add_data_relative_items = AddItems(
        state_loss_weights=state_loss_weights,
        fk_loss_weight=state_loss_weights,
        T_base2world=np.eye(4),
        joint_mask=joint_mask,
        joint_relative_pos=1 - np.eye(2),
    )
    add_non_array_items = AddItems(
        noise_type="local_joint_local_pose", to_numpy=False
    )

    state_sampling = SimpleStateSampling(
        hist_steps=max(config["hist_steps"] // 3, 1),
        pred_steps=config["pred_steps"] // 3,
    )
    action_upsample = UpSampleRobotState(
        pred_steps=config["pred_steps"],
        hist_steps=config["hist_steps"],
    )

    resize = Resize(
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
    to_tensor = ToTensor()
    projection_mat = GetProjectionMat(target_coordinate="base")
    convert_dtype = ConvertDataType(
        convert_map=dict(
            imgs=torch.float32,
            depths=torch.float32,
            image_wh=torch.float32,
            projection_mat=torch.float32,
            embodiedment_mat=torch.float32,
            joint_relative_pos=torch.float32,
            hist_robot_state=torch.float32,
            pred_robot_state=torch.float32,
        )
    )
    scale_shift = AddScaleShift(scale_shift=scale_shift)
    if mode == "training":
        item_selection = ItemSelection(
            keys=[
                "imgs",
                "depths",
                "image_wh",
                "projection_mat",
                "embodiedment_mat",
                "hist_robot_state",
                "pred_robot_state",
                "joint_scale_shift",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "joint_mask",
                "joint_relative_pos",
                "noise_type",
            ]
        )
        transforms = [
            add_data_relative_items,
            add_non_array_items,
            state_sampling,
            hand_to_gripper,
            resize,
            img_channel_flip,
            to_tensor,
            action_upsample,
            projection_mat,
            scale_shift,
            convert_dtype,
            item_selection,
        ]
    elif mode == "validation" or mode == "deploy":
        raise NotImplementedError
    return transforms


@train_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    if "egodex" not in dataset_names:
        return []
    assert mode == "training", "only support training mode"

    from robo_orchard_lab.dataset.egodex.egodex_lmdb_dataset import (
        EgoDexLmdbDataset,
    )

    transforms = build_transforms(config, mode)
    dataset = EgoDexLmdbDataset(
        paths=data_paths,
        transforms=transforms,
        lazy_init=lazy_init or mode != "training",
        dataset_name="egodex",
        reset_step=1000,
    )
    return [dataset]
