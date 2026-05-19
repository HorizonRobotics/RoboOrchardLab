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

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

DATA_TYPE = "agibot_geniesim"

g2_kinematics_config = dict(
    urdf="./urdf/G2_omnipicker_no_warnings.urdf",
    arm_joint_id=[list(range(5, 12)), list(range(20, 27))],
    arm_link_keys=[
        [
            "arm_l_link1",
            "arm_l_link2",
            "arm_l_link3",
            "arm_l_link4",
            "arm_l_link5",
            "arm_l_link6",
            "arm_l_end_link",
        ],
        [
            "arm_r_link1",
            "arm_r_link2",
            "arm_r_link3",
            "arm_r_link4",
            "arm_r_link5",
            "arm_r_link6",
            "arm_r_end_link",
        ],
    ],
    finger_keys=[
        ["gripper_l_center_link"],
        ["gripper_r_center_link"],
    ],
    head_joint_id=list(range(35, 38)),
    head_link_keys=[
        "head_link1",
        "head_link2",
        "head_link3",
    ],
    body_joint_id=list(range(5)),
    body_link_keys=[
        "body_link1",
        "body_link2",
        "body_link3",
        "body_link4",
        "body_link5",
    ],
)

cam_names = ["hand_left", "hand_right", "top_head"]


def build_transforms(config, mode, calibration, kinematics_config=None):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        CalibrationToExtrinsic,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        MoveEgoToCam,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.dataset.agibot_digit.transforms import (
        AgiBotOmniPickerKinematics,
    )

    t_base2world = np.eye(4).tolist()  # noqa: N806
    joint_mask = (
        ([True] * 7 + [False]) * 2
        + [True] * 3
        + [False, False, True, False, True]
    )

    base_joint_weights = [1] + [0] * 3 + [0] * 4
    gripper_weights = [1] + [1] * 3 + [0.1] * 4
    head_weights = [0] + [0] * 7
    body_weights = [1] + [0] * 7
    loss_weights = np.array(
        [
            [base_joint_weights] * 7
            + [gripper_weights]
            + [base_joint_weights] * 7
            + [gripper_weights]
            + [head_weights] * 3
            + [body_weights] * 5
        ]
    )
    state_loss_weights = loss_weights * 0.2
    fk_loss_weight = loss_weights * 1.8
    state_loss_weights = state_loss_weights.tolist()
    fk_loss_weight = fk_loss_weight.tolist()
    joint_scale_shift = [
        [2.205806732, -0.071631074],  # left arm j1
        [1.953144312, -0.094103515],  # left arm j2
        [2.534749031, 0.165626645],  # left arm j3
        [1.738935232, -0.741954803],  # left arm j4
        [2.395795584, -0.735441148],  # left arm j5
        [1.019364595, 0.004599452],  # left arm j6
        [1.505768418, 0.009587884],  # left arm j7
        [-0.500000000, 0.500000000],  # left gripper (from action)
        [2.205806732, -0.071631074],  # right arm j1
        [1.953144312, -0.094103515],  # right arm j2
        [2.534749031, 0.165626645],  # right arm j3
        [1.738935232, -0.741954803],  # right arm j4
        [2.395795584, -0.735441148],  # right arm j5
        [1.019364595, 0.004599452],  # right arm j6
        [1.505768418, 0.009587884],  # right arm j7
        [-0.500000000, 0.500000000],  # right gripper (from action)
        [0.009524317, 0.000949176],  # head j1
        [0.002924739, 0.000004890],  # head j2
        [0.004455832, 0.116459683],  # head j3
        [0.126622692, -0.933397055],  # body j1
        [0.084599182, 1.274401426],  # body j2
        [0.314247102, -0.346087545],  # body j3
        [0.023029560, 0.017360471],  # body j4
        [1.702966928, 0.270821095],  # body j5
    ]

    add_data_relative_items = dict(
        type=AddItems,
        T_base2world=t_base2world,
        joint_mask=joint_mask,
        joint_scale_shift=joint_scale_shift,
    )
    if mode == "training":
        add_data_relative_items.update(
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        use_master_gripper=True,
        use_master_joint=False,
        limitation=1000,
        gripper_indices=[7, 15],
    )
    dst_wh = config.get("dst_wh", (308, 252))
    dst_wh = (max(392, dst_wh[0]), max(252, dst_wh[1]))
    patch_size = config.get("patch_size", 1)
    dst_wh = tuple(x // patch_size * patch_size for x in dst_wh)
    resize = dict(
        type=Resize,
        dst_wh=dst_wh,
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
            joint_scale_shift="float32",
        ),
    )

    kinematics = dict(type=AgiBotOmniPickerKinematics, **kinematics_config)

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
                "subtask",
                "pred_mask",
                "joint_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
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
                "subtask",
                "joint_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        calib_to_ext = dict(
            type=CalibrationToExtrinsic,
            calibration=calibration,
            cam_ee_joint_indices=dict(left=6, right=14),
            **kinematics_config,
        )
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
                "remaining_actions",
                "delay_horizon",
                "joint_mask",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
            ego_to_cam,
            projection_mat,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    mode,
    **kwargs,
):
    from robo_orchard_lab.dataset.agibot_geniesim.agibot_geniesim3_ro_dataset import (
        AgibotGenieSim3RODataset,
    )
    from robo_orchard_lab.utils.build import build
    from robo_orchard_lab.utils.misc import as_sequence

    transforms = build_transforms(
        config,
        mode,
        calibration=None,
        kinematics_config=g2_kinematics_config,
    )
    return AgibotGenieSim3RODataset(
        paths=data_paths,
        cam_names=cam_names,
        target_columns=["joints", "actions"],
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        transforms=[build(x) for x in as_sequence(transforms)],
        gripper_indices=[7, 15],
        gripper_divisor=120.0,
    )


@processor_register(DATA_TYPE)
def build_processors(config, dataset_name, **kwargs):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(
        config,
        mode="deploy",
        calibration=None,
        kinematics_config=g2_kinematics_config,
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            cam_names=cam_names,
            transforms=transforms,
        )
    )
