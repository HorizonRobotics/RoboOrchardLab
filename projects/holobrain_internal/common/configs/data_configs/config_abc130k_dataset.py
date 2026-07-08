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

import uuid

import numpy as np
from dataset_factory import processor_register, train_dataset_register

DATA_TYPE = "abc130k"

dataset_config = dict(
    abc130k_dual_arm=dict(
        urdf="./urdf/abc130k_dual_arm.urdf",
        cam_names=["left", "right", "top"],
        # ABC130k extrinsics are baked in world coordinates by the packer's
        # URDF FK pass, so base==world and ego is identity too. Kept as
        # explicit np.eye(4) so `GetProjectionMat` doesn't have to guess the
        # frame downstream.
        T_base2world=np.eye(4).tolist(),
        T_base2ego=np.eye(4).tolist(),
        kinematics_config=dict(
            urdf="./urdf/abc130k_dual_arm.urdf",
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            arm_link_keys=[
                [
                    "left_link_1",
                    "left_link_2",
                    "left_link_3",
                    "left_link_4",
                    "left_link_5",
                    "left_link_6",
                ],
                [
                    "right_link_1",
                    "right_link_2",
                    "right_link_3",
                    "right_link_4",
                    "right_link_5",
                    "right_link_6",
                ],
            ],
            finger_keys=[["left_grasp_site"], ["right_grasp_site"]],
        ),
        scale_shift=[
            [0.352101557, -0.456893168],  # left_joint1
            [0.475206616, 1.345365714],  # left_joint2
            [0.476462012, 1.053014417],  # left_joint3
            [0.536535479, -0.231537762],  # left_joint4
            [0.421533844, 0.137028558],  # left_joint5
            [0.523023279, -0.413514429],  # left_joint6
            [0.406151585, 0.460548091],  # left_gripper
            [0.368711112, 0.511299894],  # right_joint1
            [0.450517247, 1.446781509],  # right_joint2
            [0.479298040, 1.138074794],  # right_joint3
            [0.506920394, -0.265883101],  # right_joint4
            [0.433787882, -0.122359766],  # right_joint5
            [0.516657226, 0.309457587],  # right_joint6
            [0.378533170, 0.413475551],  # right_gripper
        ],
        num_joint=14,
        flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "abc130k").hex[:4], 16),
    ),
)


def _build_item_selection_keys(mode):
    keys = [
        "imgs",
        "depths",
        "image_wh",
        "projection_mat",
        "embodiedment_mat",
        "hist_robot_state",
        "joint_scale_shift",
        "kinematics",
        "text",
        "joint_mask",
    ]

    if mode == "training":
        keys.extend(
            [
                "pred_robot_state",
                # Emitted by SimpleStateSampling; masks padded pred rows at
                # episode end so loss doesn't punish the model for
                # "predicting past" the trajectory.
                "pred_mask",
                "uuid",
                "subtask",
                "value",
                "fk_loss_weight",
                "state_loss_weights",
            ]
        )
    elif mode == "validation":
        keys.extend(
            [
                "pred_robot_state",
                "uuid",
                "subtask",
                "value",
            ]
        )
    elif mode == "deploy":
        # Populated by the inference server, consumed by the RTC plugin to
        # blend the previous action buffer into the new prediction.
        keys.extend(["remaining_actions", "delay_horizon"])
    return keys


def build_transforms(
    config,
    mode,
    kinematics_config,
    t_base2world,
    t_base2ego,
    scale_shift,
    num_joint,
    cam_names,
):
    # ABC130k stays on the horizon_manipulation transform pipeline (same
    # source of truth as agilex): SimpleStateSampling handles the gripper
    # column swap because ABC130k's `state` gripper is post-contact finger
    # distance while `action` is 0/1 open/close intent.
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        MultiArmKinematics,
        RandomCropPaddingResize,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.transforms import ValueSampling

    num_joint_per_arm = num_joint // 2 - 1
    joint_mask = ([True] * num_joint_per_arm + [False]) * 2

    joint_state_loss_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
        ]
    )
    state_loss_weights = (loss_weights * 0.2).tolist()
    fk_loss_weight = (loss_weights * 1.8).tolist()

    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=[6, 13],
    )

    # Match agilex: pin `dst_intrinsic` so the projection distribution is
    # decoupled from per-episode K variance and align dst_wh with the vision
    # backbone's patch size so we don't lose a stride on the edge.
    dst_wh = config.get("dst_wh", (308, 252))
    dst_wh = (max(392, dst_wh[0]), max(252, dst_wh[1]))
    patch_size = config.get("patch_size", 1)
    dst_wh = tuple(x // patch_size * patch_size for x in dst_wh)
    resize = dict(
        type=Resize,
        dst_wh=dst_wh,
        dst_intrinsic=[
            [290, 0.0, dst_wh[0] / 2, 0.0],
            [0.0, 310, dst_wh[1] / 2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
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
    scale_shift_t = dict(type=AddScaleShift, scale_shift=scale_shift)
    item_selection = dict(
        type=ItemSelection,
        keys=_build_item_selection_keys(mode),
    )
    padding_depths = np.zeros([len(cam_names), 2, 2]).tolist()
    add_padding_depths = dict(type=AddItems, depths=padding_depths)

    value_sampling = (
        dict(
            type=ValueSampling,
            norm_mode=config["value_norm_mode"],
            task_max_step=None,
        )
        if config.get("value_model_training", False)
        else None
    )

    if mode == "training":
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
            add_to_pred=True,
        )
        random_crop_padding = dict(
            type=RandomCropPaddingResize,
            range_w=(-30, 30),
            range_h=(-30, 50),
            range_scale=None,
        )
        # NB: no `ExtrinsicNoise` — the packer already runs URDF FK to bake
        # per-step extrinsics, so there's no calibration miscalibration to
        # simulate. Adding noise here would fight the reason the packer
        # produced `extrinsic_corrected` in the first place.
        transforms = [
            add_data_relative_items,
            value_sampling,
            state_sampling,
            random_crop_padding,
            add_padding_depths,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift_t,
            joint_state_noise,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "validation":
        transforms = [
            add_data_relative_items,
            value_sampling,
            state_sampling,
            add_padding_depths,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift_t,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    else:  # deploy
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            add_padding_depths,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift_t,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
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

    data_config = dataset_config[setting_type]
    transforms = build_transforms(
        config,
        mode,
        data_config["kinematics_config"],
        data_config["T_base2world"],
        data_config["T_base2ego"],
        data_config["scale_shift"],
        data_config["num_joint"],
        data_config["cam_names"],
    )
    if callable(data_paths):
        data_paths = data_paths()
    return ABC130kLmdbDataset(
        paths=data_paths,
        task_names=config.get("task_names"),
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        cam_names=data_config["cam_names"],
        reset_step=500,
        load_depth=False,
        # Required when reading sharded LMDB packs (num_steps_per_shard set).
        # Harmless for flat packs.
        hist_steps=config.get("hist_steps"),
        pred_steps=config.get("pred_steps"),
        flag=data_config.get(
            "flag",
            int(uuid.uuid5(uuid.NAMESPACE_DNS, "abc130k").hex[:4], 16),
        ),
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

    data_config = dataset_config[setting_type]
    transforms = build_transforms(
        config,
        "deploy",
        data_config["kinematics_config"],
        data_config["T_base2world"],
        data_config["T_base2ego"],
        data_config["scale_shift"],
        data_config["num_joint"],
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=False,
            valid_action_step=None,
            transforms=transforms,
            cam_names=data_config["cam_names"],
        )
    )


@processor_register(DATA_TYPE)
def build_processors(
    config,
    dataset_name,
    setting_type,
):
    return _build_processor(config, setting_type=setting_type)
