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

DATA_TYPE = "droid"


ROBOT_PROFILES = {
    "droid": dict(
        scale_shift=[
            [2.6668646335601807, 0.001893758773803711],
            [1.5998390913009644, -0.01578366756439209],
            [2.6882522106170654, 0.007451534271240234],
            [1.2947392910718918, -1.6108455210924149],
            [2.6663869619369507, -0.004207730293273926],
            [2.0163204446434975, 2.2652585729956627],
            [2.758755087852478, -0.003111720085144043],
            [-0.5, 0.5],
        ],
        kinematics_config=dict(
            urdf="./urdf/droid/franka_panda/panda.urdf",
            arm_joint_id=[list(range(7))],
            arm_link_keys=[
                [
                    "panda_link1",
                    "panda_link2",
                    "panda_link3",
                    "panda_link4",
                    "panda_link5",
                    "panda_link6",
                    "panda_link7_ee",
                ]
            ],
            finger_keys=[
                ["panda_link7_gripper_end"],
            ],
        ),
    ),
}


def get_robot_profiles():
    return ROBOT_PROFILES


def build_transforms(config, mode):
    import numpy as np
    import torch

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        IdentityTransform,
        ItemSelection,
        MoveEgoToCam,
        MultiArmKinematics,
        Resize,
        SimpleResize,
        SimpleStateSampling,
        ToTensor,
        UpSampleJointState,
    )
    from robo_orchard_lab.transforms import ValueSampling

    value_sampling = (
        dict(
            type=ValueSampling,
            norm_mode=config["value_norm_mode"],
            task_max_step=None,
        )
        if config.get("value_model_training", False)
        else None
    )

    t_base2ego = np.eye(4)
    t_base2world = np.eye(4)
    robot_profile = ROBOT_PROFILES["droid"]
    scale_shift = robot_profile["scale_shift"]

    joint_state_loss_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    num_joint = len(scale_shift) - 1
    loss_weights = np.array(
        [[joint_state_loss_weights] * num_joint + [ee_state_loss_weights]]
    )
    droid_loss_weight = 0.5
    state_loss_weights = loss_weights * 0.2 * droid_loss_weight
    fk_loss_weight = loss_weights * 1.8 * droid_loss_weight
    joint_mask = np.array([True] * num_joint + [False])

    if mode == "training":
        add_data_relative_items = AddItems(
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = AddItems(
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
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
    ego_to_cam = MoveEgoToCam()
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

    kinematics = MultiArmKinematics(**robot_profile["kinematics_config"])

    with_reference_imgs = config.get("with_reference_imgs", False)
    if with_reference_imgs:
        reference_img_dst_wh = config.get("reference_img_dst_wh", (224, 224))
        resize_reference_img = SimpleResize(
            keys="reference_imgs", dst_wh=reference_img_dst_wh
        )
    else:
        resize_reference_img = IdentityTransform()

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
                "joint_mask",
                "value",
                *(["reference_imgs"] if with_reference_imgs else []),
            ]
        )
        transforms = [
            add_data_relative_items,
            resize_reference_img,
            value_sampling,
            state_sampling,
            resize,
            to_tensor,
            joint_upsample,
            ego_to_cam,
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
                "joint_mask",
                "value",
                *(["reference_imgs"] if with_reference_imgs else []),
            ]
        )
        transforms = [
            add_data_relative_items,
            resize_reference_img,
            value_sampling,
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


def _build_dataset(
    config,
    dataset_name,
    data_paths,
    mode,
    lazy_init=True,
):
    assert mode == "training", "only support training mode"

    from robo_orchard_lab.dataset.droid.droid_lmdb_dataset import (
        DroidActionLmdbDataset,
    )

    transforms = build_transforms(config, mode)
    dataset = DroidActionLmdbDataset(
        paths=data_paths,
        transforms=transforms,
        lazy_init=lazy_init or mode != "training",
        dataset_name=dataset_name,
        min_num_step=50,
        max_num_step=1000,
        reset_step=1000,
        load_reference_img=config.get("with_reference_imgs", False),
    )
    return dataset


@train_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    mode="training",
    lazy_init=True,
):
    return _build_dataset(
        config,
        dataset_name=dataset_name,
        data_paths=data_paths,
        mode=mode,
        lazy_init=lazy_init,
    )
