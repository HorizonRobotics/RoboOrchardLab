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


data_paths = [
    "./data/droid/RAIL/success",
    "./data/droid/REAL/success",
    "./data/droid/AUTOLab/success",
    "./data/droid/GuptaLab/success",
    "./data/droid/IRIS/success",
    "./data/droid/TRI/success",
    "./data/droid/CLVR/success",
    "./data/droid/ILIAD/success",
    "./data/droid/IPRL/success",
    "./data/droid/PennPAL/success",
    "./data/droid/RPL/success",
    # f"./data/droid/WEIRD/success",
]


def build_transforms(config, mode):
    import numpy as np
    import torch

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        MultiArmKinematics,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UpSampleJointState,
    )

    t_base2ego = np.eye(4)
    t_base2world = np.eye(4)
    scale_shift = [
        [2.6668646335601807, 0.001893758773803711],
        [1.5998390913009644, -0.01578366756439209],
        [2.6882522106170654, 0.007451534271240234],
        [1.2947392910718918, -1.6108455210924149],
        [2.6663869619369507, -0.004207730293273926],
        [2.0163204446434975, 2.2652585729956627],
        [2.758755087852478, -0.003111720085144043],
        [0.5, 0.5],
    ]

    joint_state_loss_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    num_joint = len(scale_shift) - 1
    loss_weights = np.array(
        [[joint_state_loss_weights] * num_joint + [ee_state_loss_weights]]
    )
    droid_loss_weight = 0.5
    state_loss_weights = loss_weights * 0.2 * droid_loss_weight
    fk_loss_weight = loss_weights * 1.8 * droid_loss_weight

    if mode == "training":
        add_data_relative_items = AddItems(
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
        )
    else:
        add_data_relative_items = AddItems(
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
        )

    state_sampling = SimpleStateSampling(
        hist_steps=max(config["hist_steps"] // 2, 1),
        pred_steps=config["pred_steps"] // 2,
        use_master_gripper=False,
        use_master_joint=False,
        limitation=2 * 3.14,
    )

    joint_upsample = UpSampleJointState(
        pred_steps=config["pred_steps"],
        hist_steps=config["hist_steps"],
    )

    resize = Resize(
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    to_tensor = ToTensor()
    projection_mat = GetProjectionMat(target_coordinate="ego")
    convert_dtype = ConvertDataType(
        convert_map=dict(
            imgs=torch.float32,
            depths=torch.float32,
            image_wh=torch.float32,
            projection_mat=torch.float32,
            embodiedment_mat=torch.float32,
        )
    )

    kinematics = MultiArmKinematics(
        urdf="./urdf/franka_description/panda.urdf",
        arm_joint_id=[list(range(7))],
        arm_link_keys=[
            [
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7",
            ]
        ],
        finger_keys=[
            ["left_inner_finger", "right_inner_finger"],
        ],
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
                "kinematics",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "subtask",
            ]
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            joint_upsample,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "validation":
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
                "kinematics",
                "text",
                "uuid",
                "subtask",
            ]
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            joint_upsample,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        raise NotImplementedError
    return transforms


def build_datasets(config, dataset_names, mode, lazy_init=True):
    if "droid" not in dataset_names:
        return []
    assert mode == "training", "only support training mode"

    from robo_orchard_lab.dataset.droid.droid_lmdb_dataset import (
        DroidActionLmdbDataset,
    )

    transforms = build_transforms(config, mode)
    dataset = DroidActionLmdbDataset(
        paths=data_paths,
        transforms=transforms,
        lazy_init=lazy_init or mode != "training",
        dataset_name="droid",
        min_num_step=50,
        max_num_step=1000,
        reset_step=1000,
    )
    return [dataset]
