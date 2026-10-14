# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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


from projects.holobrain_0_5.configs.robot_specs import get_robot_spec
from projects.holobrain_0_5.configs.transforms import (
    AddItems,
    ConvertDataType,
    ImageChannelFlip,
    MultiArmKinematics,
    Resize,
)

__all__ = ["build_datasets", "build_processors", "build_transforms"]

DATA_TYPE = "robotwin"
dataset_config = dict(
    aloha_v2=dict(
        robot_type="robotwin_aloha",
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.5, 0.5],
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.5, 0.5],
        ],
        cam_names=[
            "front_camera",
            "left_camera",
            "right_camera",
            "head_camera",
        ],
    ),
    ur5_wsg=dict(
        robot_type="robotwin_ur5_wsg",
        T_base2world=[
            [1, 0, 0, 0],
            [0, 1, 0, -0.65],
            [0, 0, 1, 0.65],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [2.400281548500061, -0.1310516595840454],
            [1.445511817932129, -1.445511817932129],
            [2.16847026348114, -0.23492777347564697],
            [1.7424615025520325, -0.007538259029388428],
            [2.8101450204849243, 0.15472495555877686],
            [2.9653799533843994, 0.02583003044128418],
            [0.5, 0.5],
            [2.400281548500061, -0.1310516595840454],
            [1.445511817932129, -1.445511817932129],
            [2.16847026348114, -0.23492777347564697],
            [1.7424615025520325, -0.007538259029388428],
            [2.8101450204849243, 0.15472495555877686],
            [2.9653799533843994, 0.02583003044128418],
            [0.5, 0.5],
        ],
        cam_names=["left_camera", "right_camera", "head_camera"],
    ),
)


def build_transforms(config, mode, setting_type):
    """Build action transforms for training, validation or deployment."""
    import numpy as np

    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddScaleShift,
        GetProjectionMat,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    setting = dataset_config[setting_type]
    t_base2world = setting["T_base2world"]
    scale_shift = setting["scale_shift"]
    spec = get_robot_spec(setting["robot_type"])
    urdf = spec["urdf"]
    num_joint = spec["num_joint"]
    joint_mask = spec["joint_mask"]
    kinematics_config = dict(spec["kinematics"], urdf=urdf)
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
    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
            joint_mask=joint_mask,
            arm_length=spec["arm_length"],
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
            arm_length=spec["arm_length"],
        )
    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
        strict_depth=config["with_depth"],
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")
    convert_dtype = dict(
        type=ConvertDataType,
        convert_map=dict(
            arm_length="float32",
            imgs="float32",
            depths="float32",
            image_wh="float32",
            projection_mat="float32",
            embodiedment_mat="float32",
        ),
    )
    kinematics = dict(type=MultiArmKinematics, **kinematics_config)
    scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)
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
                "arm_length",
                "fk_loss_weight",
                "state_loss_weights",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
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
                "arm_length",
                "text",
                "uuid",
                "joint_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
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
                "arm_length",
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
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return [x for x in transforms if x is not None]


def build_datasets(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode="training",
    lazy_init=True,
):
    """Build an LMDB reader using the selected body and camera order."""
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    transforms = build_transforms(config, mode, setting_type)
    return RoboTwinLmdbDataset(
        paths=data_paths,
        task_names=config.get("task_names"),
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        load_image=config.get("load_image", True),
        load_depth=config.get("load_depth", True),
        cam_names=dataset_config[setting_type]["cam_names"],
        reset_step=1000,
    )


def _build_processor(config, setting_type):
    from robo_orchard_lab.models.holobrain_0_5 import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(config, "deploy", setting_type)
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            transforms=transforms,
            cam_names=dataset_config[setting_type]["cam_names"],
        )
    )


def build_processors(config, dataset_name, setting_type):
    """Build the matching deployment processor for a dataset selection."""
    return _build_processor(config, setting_type=setting_type)
