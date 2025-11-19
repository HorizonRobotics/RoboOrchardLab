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

logger = logging.getLogger(__name__)


dataset_config = dict(
    behavior1k=dict(
        kinematics_config=dict(
            urdf="./urdf/r1_pro_with_gripper.urdf",
            # torso_joint_id=list(range(6, 10)),
            left_arm_joint_id=list(range(10, 19)),  # including 2 gipper
            right_arm_joint_id=list(range(19, 28)),  # including 2 gipper
            left_arm_link_keys=[
                "left_arm_base_link",
                "left_arm_link1",
                "left_arm_link2",
                "left_arm_link3",
                "left_arm_link4",
                "left_arm_link5",
                "left_arm_link6",
                "left_arm_link7",
                "left_gripper_link",
                # "left_realsense_link",
            ],
            left_finger_keys=[
                "left_gripper_finger_link1",
                # "left_gripper_finger_link2",
            ],
            right_arm_link_keys=[
                "right_arm_base_link",
                "right_arm_link1",
                "right_arm_link2",
                "right_arm_link3",
                "right_arm_link4",
                "right_arm_link5",
                "right_arm_link6",
                "right_arm_link7",
                "right_gripper_link",
                # "right_realsense_link",
            ],
            right_finger_keys=[
                "right_gripper_finger_link1",
                # "right_gripper_finger_link2",
            ],
            # use_ee_state=True,
        ),
        scale_shift=[
            # base
            # [ 0.7500, 0.0000],
            # [ 0.7500, 0.0000],
            # [ 1.0000, 0.0000],
            # torso
            [1.4836, 0.3491],
            [2.6616, -0.1309],
            [1.7017, -0.1309],
            [3.0543, 0.0000],
            # left arm
            [2.8798, -1.5708],
            [1.6580, 1.4835],
            [2.3562, 0.0000],
            [1.2218, -0.8726],
            [2.3562, 0.0000],
            [1.0472, 0.0000],
            [1.5708, 0.0000],
            [1.0000, 0.0000],
            # [ 1.0000, 0.0000],
            # right arm
            [2.8798, -1.5708],
            [1.6580, -1.4835],
            [2.3562, 0.0000],
            [1.2218, -0.8726],
            [2.3562, 0.0000],
            [1.0472, 0.0000],
            [1.5708, 0.0000],
            [1.0000, 0.0000],
            # [ 0.0250, 0.0250],
            # [ 0.0250, 0.0250],
        ],
        data_paths=[
            "/work/bucket/dataset/behavior1k_with_moving_subtask/task_0000_lmdb",
        ],
        num_joint=28,
        dst_wh=[476, 476],
        cam_names=[
            "left_wrist",
            "right_wrist",
            "head",
        ],
        use_base_qvel_as_input=True,
    )
)


def build_transforms(
    config, mode, kinematics_config, scale_shift, num_joint, dst_wh
):
    from robo_orchard_lab.dataset.behavior.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        R1ProDualArmKinematics,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    loss_weights = np.array([[joint_state_loss_weights] * 20]).tolist()

    add_data_relative_items = dict(
        type=AddItems,
        state_loss_weights=loss_weights,
        fk_loss_weight=loss_weights,
        T_base2ego=np.eye(4).tolist(),
        T_base2world=np.eye(4).tolist(),
    )

    resize = dict(type=Resize, dst_wh=dst_wh)

    to_tensor = dict(type=ToTensor)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="base")
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
    kinematics = dict(type=R1ProDualArmKinematics, **kinematics_config)
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
            "hist_joint_state",
            "pred_joint_state",
            "mobile_traj",
            # "hist_ee_state",
            "joint_relative_pos",
            "joint_scale_shift",
            "kinematics",
            "text",
            "uuid",
            "pred_mask",
            "subtask",
            "state_loss_weights",
            "fk_loss_weight",
            "T_world2cam",
            "T_cam2base",
            "intrinsic",
        ],
    )

    # state_sampling = dict(
    #    type=SimpleStateSampling,
    #    hist_steps=config["hist_steps"],
    #    pred_steps=config["pred_steps"],
    # )

    transforms = [
        add_data_relative_items,
        # state_sampling,
        resize,
        to_tensor,
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
        transforms.insert(1, state_sampling)
    elif mode == "validation":
        state_sampling = dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            mode="validation",
        )

        transforms.insert(1, state_sampling)
    elif mode == "deploy":
        state_sampling = dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            mode="deploy",
        )
        transforms.insert(1, state_sampling)

        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms.append(unsqueeze_batch)

    return transforms


def build_datasets(
    config,
    dataset_names,
    lazy_init=True,
    mode="training",
):
    """Build Behavior datasets for training."""
    assert mode == "training", "only support training mode"
    from robo_orchard_lab.dataset.behavior.behavior_lmdb_dataset import (
        BehaviorLmdbDataset,
    )

    datasets = []
    for _, data_config in dataset_config.items():
        transforms = build_transforms(
            config,
            mode,
            data_config["kinematics_config"],
            data_config["scale_shift"],
            data_config["num_joint"],
            data_config["dst_wh"],
        )

        dataset = BehaviorLmdbDataset(
            paths=data_config["data_paths"],
            transforms=transforms,
            lazy_init=lazy_init or mode != "training",
        )

        datasets.append(dataset)

    return datasets


def build_processors(config, dataset_names):
    from robo_orchard_lab.models.sem_modules import (
        SEMProcessor,
        SEMProcessorCfg,
    )

    processors = {}
    for dataset_name, data_config in dataset_config.items():
        transforms = build_transforms(
            config,
            "deploy",
            data_config["kinematics_config"],
            data_config["scale_shift"],
            data_config["num_joint"],
            data_config["dst_wh"],
        )

        processor = SEMProcessor(
            SEMProcessorCfg(
                load_image=True,
                load_depth=config["with_depth"],
                valid_action_step=None,
                transforms=transforms,
                cam_names=data_config["cam_names"],
            )
        )
        processors[dataset_name] = processor

    return processors
