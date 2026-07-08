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
import torch
from dataset_factory import train_dataset_register

logger = logging.getLogger(__name__)

DATA_TYPE = "agibot"


kinematics_config = dict(
    agibot=dict(
        urdf="./urdf/G1_120s_dual.urdf",
        arm_joint_id=[
            list(range(2, 9)),  # Joint1_l - Joint7_l in the URDF chain
            list(range(17, 24)),  # Joint1_r - Joint7_r in the URDF chain
            [1, 0],  # joint_body_pitch, joint_lift_body in the URDF chain
        ],
        arm_link_keys=[
            [
                "Link1_l",
                "Link2_l",
                "Link3_l",
                "Link4_l",
                "Link5_l",
                "Link6_l",
                "Link7_l",
            ],
            [
                "Link1_r",
                "Link2_r",
                "Link3_r",
                "Link4_r",
                "Link5_r",
                "Link6_r",
                "Link7_r",
            ],
            ["link-pitch_body", "link-up-down_body"],
        ],
        finger_keys=[["gripper_center"], ["right_gripper_center"], []],
    )
)

agibot_scale_shift = [
    # Left arm joints (7 joints)
    [0.807397993, -1.075758200],  # Joint1_l
    [0.519545991, 0.628004746],  # Joint2_l
    [0.827265625, 0.755998523],  # Joint3_l
    [0.431041460, -0.748906466],  # Joint4_l
    [0.357815969, 0.392597735],  # Joint5_l
    [0.389363049, 1.321284795],  # Joint6_l
    [0.477344665, 0.298801437],  # Joint7_l
    [-33.615918755, 54.628037467],  # left_gripper
    # Right arm joints (7 joints)
    [1.056550615, 0.897950653],  # Joint1_r
    [0.610640659, -0.593406163],  # Joint2_r
    [0.892552863, -0.728802693],  # Joint3_r
    [0.413018960, 0.736117755],  # Joint4_r
    [0.567352174, -0.411271768],  # Joint5_r
    [0.313294437, -1.254856470],  # Joint6_r
    [0.490895931, -0.434084028],  # Joint7_r
    [-34.681226171, 61.714828429],  # right_gripper
    # Head joints (2 joints)
    [0.049485552, -0.027264017],  # joint_head_pitch
    [0.077976995, 0.451097382],  # joint_head_yaw
    # Body joints (2 joints)
    [0.082625759, 0.430035826],  # joint_body_pitch
    [0.076923081, 0.274950588],  # joint_lift_body
]



def build_transforms(config, reference_img_path):
    from robo_orchard_lab.dataset.agibot.transforms import (
        AddItems,
        ConvertDataType,
        GetProjectionMat,
        IdentityTransform,
        ItemSelection,
        JointSelection,
        LoadReferenceImages,
        MoveEgoToCam,
        Resize,
        SimpleResize,
        SimpleStateSampling,
        TextAug,
        ToTensor,
        UpSampleJointState,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        MultiArmKinematics,
    )
    from robo_orchard_lab.transforms import ValueSampling

    with_reference_imgs = reference_img_path is not None and config.get(
        "with_reference_imgs", False
    )
    if with_reference_imgs:
        load_reference_img = LoadReferenceImages(reference_img_path)
        reference_img_dst_wh = config.get("reference_img_dst_wh", (224, 224))
        resize_reference_img = SimpleResize(
            keys="reference_imgs", dst_wh=reference_img_dst_wh
        )
    else:
        load_reference_img = IdentityTransform()
        resize_reference_img = IdentityTransform()

    value_sampling = dict(
        type=ValueSampling,
        norm_mode=config["value_norm_mode"],
        task_max_step=None,
    ) if config.get("value_model_training", False) else None

    state_sampling = SimpleStateSampling(
        hist_steps=config["hist_steps"] // 3 + 1,
        pred_steps=config["pred_steps"] // 3 + 1,
        gripper_indices=[7, 15],  # AgiBot gripper indices
    )

    joint_upsample = UpSampleJointState(
        pred_steps=config["pred_steps"],
        hist_steps=config["hist_steps"],
    )

    dst_wh = config.get("dst_wh", (308, 252))
    resize = Resize(dst_wh=dst_wh)

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
            pred_mask=torch.bool,
        )
    )

    kinematics = MultiArmKinematics(**kinematics_config["agibot"])

    item_selection = ItemSelection(
        keys=[
            "imgs",
            "depths",
            "image_wh",
            "projection_mat",
            "embodiedment_mat",
            "hist_robot_state",
            # "hist_ee_state",
            "pred_robot_state",
            "joint_relative_pos",
            "joint_scale_shift",
            "kinematics",
            "text",
            "uuid",
            "pred_mask",
            "subtask",
            "state_loss_weights",
            "fk_loss_weight",
            "joint_mask",
            "value",
            *(["reference_imgs"] if with_reference_imgs else []),
        ]
    )

    # Joint selection to filter to dual-arm joints only (16 out of 20)
    joint_selection = JointSelection(selection_mode="no_head")

    base_joint_weights = [
        1,
    ] + [0.0] * 7  # Regular arm joints: pos > rot
    lift_weights = [
        1,
    ] + [0.0] * 7  # Lift joints: same as base
    gripper_weights = [
        1,
        1,
        1,
        1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]  # Gripper joints: same as base
    l1, l2 = 0.2, 1.8
    loss_weight = np.array(
        [
            [base_joint_weights] * 7  # left arm (7)
            + [gripper_weights] * 1  # left gripper (1)
            + [base_joint_weights] * 7  # right arm (7)
            + [gripper_weights] * 1  # right gripper (1)
            + [lift_weights] * 2  # lift joints (2)
        ]
    )  # 1, num_joint, 8
    joint_mask = ([True] * 7 + [False]) * 2 + [True, False]

    add_data_relative_items = AddItems(
        state_loss_weights=loss_weight * l1,
        fk_loss_weight=loss_weight * l2,
        T_base2ego=np.eye(4),
        T_base2world=np.eye(4),
        joint_mask=np.array(joint_mask),
        joint_scale_shift=np.array(agibot_scale_shift, dtype=np.float32),
    )

    text_aug = TextAug()

    return [
        add_data_relative_items,
        load_reference_img,
        resize_reference_img,
        value_sampling,
        state_sampling,
        resize,
        to_tensor,
        joint_upsample,
        ego_to_cam,
        projection_mat,
        convert_dtype,
        joint_selection,  # Filter joint states and scale_shift to no-head
        kinematics,
        text_aug,
        item_selection,
    ]


@train_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    instruction_path,
    lazy_init=True,
    lmdb_kwargs=None,
    mode="training",
    reference_img_path=None,
):
    """Build AgiBot datasets for training."""
    assert mode == "training", "only support training mode"
    from robo_orchard_lab.dataset.agibot.agibot_lmdb_dataset import (
        AgiBotLmdbDataset,
    )
    from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
        InstructionReader,
    )

    if callable(data_paths):
        data_paths = data_paths()

    transforms = build_transforms(config, reference_img_path)
    instruction_reader = InstructionReader(paths=instruction_path)
    dataset = AgiBotLmdbDataset(
        paths=data_paths,
        lazy_init=lazy_init,
        transforms=transforms,
        cam_names=None,
        dataset_name=dataset_name,
        instruction_reader=instruction_reader,
        reset_step=500,
    )
    return dataset
