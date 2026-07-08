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
from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

logger = logging.getLogger(__name__)

DATA_TYPE = "behavior"


BEHAVIOR1K_CONFIG = dict(
    kinematics_config=dict(
        urdf="./urdf/r1_pro_with_gripper.urdf",
        arm_joint_id=[
            list(range(6, 10)),
            list(range(10, 17)),
            list(range(19, 26)),
        ],
        arm_link_keys=[
            [
                "torso_link1",
                "torso_link2",
                "torso_link3",
                "torso_link4",
            ],
            [
                "left_arm_link1",
                "left_arm_link2",
                "left_arm_link3",
                "left_arm_link4",
                "left_arm_link5",
                "left_arm_link6",
                "left_arm_link7",
            ],
            [
                "right_arm_link1",
                "right_arm_link2",
                "right_arm_link3",
                "right_arm_link4",
                "right_arm_link5",
                "right_arm_link6",
                "right_arm_link7",
            ],
        ],
        finger_keys=[
            [],
            ["left_gripper_finger_link1"],
            ["right_gripper_finger_link1"],
        ],
        arm_connection_joint_indices=[3, 0, 0],
    ),
    scale_shift=[
        # torso
        [0.6077, 1.0566],
        [1.0677, -1.4653],
        [0.8349, -0.7062],
        # [0.0003, 0.0000],
        [1.000, 0.0000],
        # left arm
        [1.0423, -0.6566],
        [0.3783, 0.2038],
        [1.4073, -0.4143],
        [0.8466, -0.7800],
        [1.6147, 0.5790],
        [0.9966, 0.0484],
        [1.3972, -0.1692],
        [0.0443, 0.0557],
        # right arm
        [0.9649, -0.7043],
        [0.3807, -0.2062],
        [1.1241, 0.1019],
        [0.8435, -0.7689],
        [1.5155, -0.3252],
        [0.9975, 0.0493],
        [1.3175, 0.0838],
        [0.0463, 0.0537],
    ],
    cam_names=[
        "left_wrist",
        "right_wrist",
        "head",
    ],
)


def build_transforms(config, mode, kinematics_config, scale_shift):
    from robo_orchard_lab.dataset.behavior.transforms import (
        AddItems,
        AddScaleShift,
        CameraMask,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        MultiArmKinematics,
    )

    joint_mask = (
        # torso
        [True] * 4
        +
        # left arm
        [True] * 7
        + [False]
        +
        # right arm
        [True] * 7
        + [False]
    )
    add_joint_mask = dict(
        type=AddItems,
        joint_mask=joint_mask,
    )

    resize = dict(type=Resize, dst_wh=config.get("dst_wh", (336, 336)))
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
        ),
    )
    kinematics = dict(type=MultiArmKinematics, **kinematics_config)
    scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)

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
            "mobile_traj",
            "joint_relative_pos",
            "joint_scale_shift",
            "kinematics",
            "text",
            "subtask_text",
            "uuid",
            "pred_mask",
            "state_loss_weights",
            "fk_loss_weight",
            "joint_mask",
        ],
    )

    transforms = [
        add_joint_mask,
        resize,
        to_tensor,
        ego_to_cam,
        projection_mat,
        scale_shift,
        convert_dtype,
        kinematics,
        item_selection,
    ]

    if mode == "training":
        state_sampling = dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            mode="training",
        )
        transforms.insert(0, state_sampling)

        joint_noise = dict(
            type=JointStateNoise, noise_range=[-0.02, 0.02], add_to_pred=True
        )
        transforms.insert(1, joint_noise)

        camera_mask = dict(type=CameraMask, max_masks=3)
        transforms.insert(2, camera_mask)

        joint_state_loss_weights = [1.0, 0, 0, 0, 0, 0, 0, 0]
        ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
        loss_weights = np.array([[joint_state_loss_weights] * 20])
        ee_indices = [11, 19]
        loss_weights[:, ee_indices] = ee_state_loss_weights
        add_loss_weight = dict(
            type=AddItems,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
        )
        transforms.append(add_loss_weight)

    elif mode == "validation":
        state_sampling = dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            mode="validation",
        )

        transforms.insert(0, state_sampling)
    elif mode == "deploy":
        state_sampling = dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            mode="deploy",
        )
        transforms.insert(0, state_sampling)

        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms.append(unsqueeze_batch)

    return transforms


@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    lazy_init=True,
    mode="training",
):
    """Build Behavior datasets for training."""
    assert mode == "training", "only support training mode"

    import uuid

    from robo_orchard_lab.dataset.behavior.behavior_lmdb_dataset import (
        BehaviorLmdbDataset,
    )

    transforms = build_transforms(
        config,
        mode,
        BEHAVIOR1K_CONFIG["kinematics_config"],
        BEHAVIOR1K_CONFIG["scale_shift"],
    )

    return BehaviorLmdbDataset(
        paths=data_paths,
        transforms=transforms,
        lazy_init=lazy_init or mode != "training",
        reset_step=1000,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        dataset_name=dataset_name,
        flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "behavior").hex[:4], 16),
    )


@processor_register(DATA_TYPE)
def build_processors(config, dataset_name, **kwargs):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(
        config,
        "deploy",
        BEHAVIOR1K_CONFIG["kinematics_config"],
        BEHAVIOR1K_CONFIG["scale_shift"],
    )

    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            transforms=transforms,
            cam_names=BEHAVIOR1K_CONFIG["cam_names"],
        )
    )
