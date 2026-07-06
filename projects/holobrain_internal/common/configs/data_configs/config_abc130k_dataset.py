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

import numpy as np
from dataset_factory import processor_register, train_dataset_register

DATA_TYPE = "abc130k"
ABC130K_HAS_DEPTH = False


dataset_config = dict(
    abc130k_dual_arm=dict(
        kinematics_config=dict(
            urdf="./urdf/abc130k_dual_arm.urdf",  # ABC130K_YAM_DUAL_ARM_URDF,
            left_arm_link_keys=[
                "left_link_1",
                "left_link_2",
                "left_link_3",
                "left_link_4",
                "left_link_5",
                "left_link_6",
            ],
            right_arm_link_keys=[
                "right_link_1",
                "right_link_2",
                "right_link_3",
                "right_link_4",
                "right_link_5",
                "right_link_6",
            ],
            left_finger_keys=[
                "left_grasp_site",
            ],
            right_finger_keys=[
                "right_grasp_site",
            ],
            left_arm_joint_id=list(range(6)),
            right_arm_joint_id=list(range(8, 14)),
        ),
        # ABC130k extrinsics are already in world coordinates (see the URDF
        # FK bake in the packer), so base==world and ego is identity too.
        # Kept explicit so downstream `GetProjectionMat` can consume them
        # without every branch guessing the frame.
        T_base2world=np.eye(4).tolist(),
        T_base2ego=np.eye(4).tolist(),
        scale_shift=[[1.0, 0.0]] * 14,
        num_joint=14,
        cam_names=["top", "left", "right"],
    ),
)


def _build_convert_map(with_depth):
    convert_map = dict(
        imgs="float32",
        image_wh="float32",
        projection_mat="float32",
        embodiedment_mat="float32",
    )
    if with_depth:
        convert_map["depths"] = "float32"
    return convert_map


def _build_item_selection_keys(mode, with_depth):
    keys = [
        "imgs",
        "image_wh",
        "projection_mat",
        "embodiedment_mat",
        "hist_robot_state",
        "joint_scale_shift",
        "kinematics",
        "text",
        "joint_mask",
    ]
    if with_depth:
        keys.insert(1, "depths")

    if mode in ["training", "validation"]:
        keys.extend(
            [
                "pred_robot_state",
                # Emitted by horizon.SimpleStateSampling; masks padded pred
                # rows at episode end so loss doesn't punish the model for
                # not "predicting past" the trajectory.
                "pred_mask",
                "uuid",
                "value",
            ]
        )
    if mode == "training":
        keys.extend(["fk_loss_weight", "state_loss_weights"])
    return keys


def build_transforms(
    config,
    mode,
    kinematics_config,
    t_base2world,
    t_base2ego,
    scale_shift,
    num_joint,
):
    # ABC130k uses horizon's SimpleStateSampling because gripper `state` is the
    # post-contact finger distance, while the commanded `action` is 0/1 open/
    # close intent. horizon's variant supports swapping master_joint_state into
    # gripper columns so BC targets can actually close on new objects.
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        SimpleStateSampling,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        DualArmKinematics,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        Resize,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.transforms import ValueSampling

    with_depth = config.get("with_depth", True) and ABC130K_HAS_DEPTH

    value_sampling = (
        dict(
            type=ValueSampling,
            norm_mode=config["value_norm_mode"],
            task_max_step=None,
        )
        if config.get("value_model_training", False)
        else None
    )

    num_joint_per_arm = num_joint // 2 - 1
    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    ee_state_loss_weights = [1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
        ]
    ).tolist()
    joint_mask = ([True] * num_joint_per_arm + [False]) * 2

    # AddItems injects the per-dataset constants that used to live on the
    # dataset object (T_base2world / T_base2ego) plus the loss weights
    # consumed by the trainer. horizon's config_agilex_dataset.py does the
    # same thing — dataset stays generic, transforms carry the constants.
    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            T_base2ego=t_base2ego,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            T_base2ego=t_base2ego,
            joint_mask=joint_mask,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        # ABC130k joint layout: [L_arm(6), L_gripper, R_arm(6), R_gripper].
        # Only override the gripper columns with master (action) values; arm
        # pred stays from the real state signal.
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=[6, 13],
        # Keep robotwin's pred timing (pred_state = state[step+1:step+1+K]).
        # horizon's default 1e-3 would skip static frames forward, which would
        # change arm pred timing across the whole dataset.
        static_threshold=0,
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="world")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=_build_convert_map(with_depth),
    )

    kinematics = dict(type=DualArmKinematics, **kinematics_config)

    scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)
    item_selection = dict(
        type=ItemSelection,
        keys=_build_item_selection_keys(mode, with_depth),
    )

    transforms = [
        add_data_relative_items,
        value_sampling,
        state_sampling,
        resize,
        img_channel_flip,
        to_tensor,
        ego_to_cam,
        projection_mat,
        scale_shift,
    ]

    if mode == "training":
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
        )
        transforms.append(joint_state_noise)

    transforms.extend([convert_dtype, kinematics, item_selection])

    if mode == "deploy":
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms.append(unsqueeze_batch)
    return transforms


def _build_dataset(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode,
    lazy_init=True,
):
    from robo_orchard_lab.dataset.abc130k.abc130k_lmdb_dataset import (
        ABC130kLmdbDataset,
    )

    transforms = build_transforms(
        config,
        mode,
        dataset_config[setting_type]["kinematics_config"],
        dataset_config[setting_type]["T_base2world"],
        dataset_config[setting_type]["T_base2ego"],
        dataset_config[setting_type]["scale_shift"],
        dataset_config[setting_type]["num_joint"],
    )
    return ABC130kLmdbDataset(
        paths=data_paths,
        task_names=config.get("task_names"),
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        cam_names=dataset_config[setting_type]["cam_names"],
        reset_step=1000,
        load_depth=config.get("with_depth", True) and ABC130K_HAS_DEPTH,
        # Required when reading sharded LMDB packs (num_steps_per_shard set).
        # Harmless for flat packs.
        hist_steps=config.get("hist_steps"),
        pred_steps=config.get("pred_steps"),
    )


@train_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode="training",
    lazy_init=True,
):
    return _build_dataset(
        config,
        dataset_name=dataset_name,
        data_paths=data_paths,
        setting_type=setting_type,
        mode=mode,
        lazy_init=lazy_init,
    )


def _build_processor(config, setting_type):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(
        config,
        "deploy",
        dataset_config[setting_type]["kinematics_config"],
        dataset_config[setting_type]["T_base2world"],
        dataset_config[setting_type]["T_base2ego"],
        dataset_config[setting_type]["scale_shift"],
        dataset_config[setting_type]["num_joint"],
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"] and ABC130K_HAS_DEPTH,
            valid_action_step=None,
            transforms=transforms,
            cam_names=dataset_config[setting_type]["cam_names"],
        )
    )


@processor_register(DATA_TYPE)
def build_processors(
    config,
    dataset_name,
    setting_type,
):
    return _build_processor(config, setting_type=setting_type)
