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

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

DATA_TYPE = "robodojo"


dataset_config = dict(
    arx_x5a=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/arx_x5a/robotwin2_dual_arm_arx_x5a.urdf",
            arm_link_keys=[
                [
                    "left_link1",
                    "left_link2",
                    "left_link3",
                    "left_link4",
                    "left_link5",
                    "left_link6_ee",
                ],
                [
                    "right_link1",
                    "right_link2",
                    "right_link3",
                    "right_link4",
                    "right_link5",
                    "right_link6_ee",
                ],
            ],
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            finger_keys=[
                ["left_link6_gripper_end"],
                ["right_link6_gripper_end"],
            ],
            arm_connection_joint_indices=[6, 6],
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.45],
            [0, 0, 1, 0.765],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [1.620033741, 0.133668900],
            [1.656651929, 1.398018405],
            [1.960509863, 1.892008934],
            [2.180836082, -0.083378673],
            [1.649001718, 0.161324859],
            [3.076746464, 0.059907198],
            [0.500000000, 0.500000000],
            [1.582818449, 0.090382993],
            [1.828920946, 1.550825015],
            [2.220252730, 2.032503866],
            [2.532093048, -0.003369808],
            [3.105193019, 0.038864970],
            [3.043393612, -0.005282640],
            [0.500000000, 0.500000000],
        ],
        num_joint=14,
        cam_names=["cam_left_wrist", "cam_right_wrist", "cam_head"],
    ),
)


def build_transforms(
    config,
    mode,
    kinematics_config,
    t_base2world,
    scale_shift,
    num_joint,
):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        ConvertDataType,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        MultiArmKinematics,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
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

    num_joint_per_arm = num_joint // 2 - 1
    joint_mask = ([True] * num_joint_per_arm + [False]) * 2

    if mode == "training":
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

        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
            joint_mask=joint_mask,
            depths=np.zeros(
                [3, 2, 2]
            ).tolist(),  # 3 cameras, fake size: [2, 2]
            joint_scale_shift=scale_shift,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
            depths=np.zeros(
                [3, 2, 2]
            ).tolist(),  # 3 cameras, fake size: [2, 2]
            joint_scale_shift=scale_shift,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=[6, 13],
        limitation=5,
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
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
                "kinematics",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "joint_mask",
                "value",
            ],
        )
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
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
            joint_state_noise,
            convert_dtype,
            kinematics,
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
                "kinematics",
                "text",
                "uuid",
                "joint_mask",
                "value",
            ],
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
            convert_dtype,
            kinematics,
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
                "joint_scale_shift",
                "kinematics",
                "text",
                "joint_mask",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    if (config.get("memoryvla") or {}).get("enable", False):
        # The memory bank keys on (episode, frame). `uuid` is already
        # whitelisted; `step_index` is produced by the dataset but dropped
        # here, so add it back -- and only with the port switched on, so a
        # baseline run sees exactly the batch it saw before.
        item_selection["keys"] = list(item_selection["keys"]) + ["step_index"]

    return transforms


@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode="training",
    lazy_init=True,
):
    from robo_orchard_lab.dataset.robodojo.robodojo_lmdb_dataset import (
        RoboDojoLmdbDataset,
    )

    transforms = build_transforms(
        config,
        mode,
        dataset_config[setting_type]["kinematics_config"],
        dataset_config[setting_type]["T_base2world"],
        dataset_config[setting_type]["scale_shift"],
        dataset_config[setting_type]["num_joint"],
    )
    return RoboDojoLmdbDataset(
        paths=data_paths,
        task_names=config.get("task_names"),
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        cam_names=dataset_config[setting_type]["cam_names"],
        reset_step=1000,
        hist_steps=config.get("hist_steps"),
        pred_steps=config.get("pred_steps"),
    )


@processor_register(DATA_TYPE)
def build_processors(
    config,
    dataset_name,
    setting_type,
):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(
        config,
        "deploy",
        dataset_config[setting_type]["kinematics_config"],
        dataset_config[setting_type]["T_base2world"],
        dataset_config[setting_type]["scale_shift"],
        dataset_config[setting_type]["num_joint"],
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            transforms=transforms,
            cam_names=dataset_config[setting_type]["cam_names"],
        )
    )
