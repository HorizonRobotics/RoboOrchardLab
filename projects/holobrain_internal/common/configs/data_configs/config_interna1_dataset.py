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

from dataset_factory import processor_register, train_dataset_register

DATA_TYPE = "interna1"


def get_dataset_lmdb_config():
    dataset_lmdb_config = dict(
        interna1_arx_lift2=dict(
            urdf="./urdf/InternData-A1_urdf/ARX_Lift2_fix/lift.urdf",
            cam_names=["hand_left", "hand_right", "head"],
            robot_type="ARX Lift-2",
            task_names=None,
            load_extrinsic=True,
        ),
        interna1_agile_split_aloha=dict(
            urdf="./urdf/InternData-A1_urdf/AgileX_Split_Aloha_piper100/split_aloha_mid_360_with_piper.sanitized.urdf",
            cam_names=["hand_left", "hand_right", "head"],
            robot_type="AgileX Split Aloha",
            task_names=None,
            load_extrinsic=True,
        ),
        interna1_genieg1=dict(
            urdf="./urdf/InternData-A1_urdf/G1_120s/G1_120s.urdf",
            cam_names=["hand_left", "hand_right", "head"],
            robot_type="Genie-1",
            task_names=None,
            load_extrinsic=True,
        ),
    )

    return dataset_lmdb_config


def build_lmdb_transforms(
    config,
    mode,
    urdf,
    robot_type,
    calibration=None,
    depth_restore=False,
    do_calib_to_ext=False,
):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        CalibrationToExtrinsic,
        ConvertDataType,
        DepthRestoration,
        GetProjectionMat,
        IdentityTransform,
        ItemSelection,
        MoveEgoToCam,
        MultiArmKinematics,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    if depth_restore:
        depth_restoration = dict(type=DepthRestoration)
    else:
        depth_restoration = dict(type=IdentityTransform)

    t_base2ego = np.eye(4).tolist()  # noqa: N806
    t_base2world = np.eye(4).tolist()  # noqa: N806
    joint_mask = ([True] * 6 + [False]) * 2

    joint_state_loss_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * 6
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * 6
            + [ee_state_loss_weights]
        ]
    )
    state_loss_weights = loss_weights * 0.2
    fk_loss_weight = loss_weights * 1.8
    state_loss_weights = state_loss_weights.tolist()
    fk_loss_weight = fk_loss_weight.tolist()

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
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
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
    arx_kinematics_config = dict(
        urdf=urdf,
        arm_joint_id=[list(range(6, 12)), list(range(14, 20))],
        arm_link_keys=[
            [
                "L_arm_link11",
                "L_arm_link12",
                "L_arm_link13",
                "L_arm_link14",
                "L_arm_link15",
                "L_arm_link16",
            ],
            [
                "R_arm_link21",
                "R_arm_link22",
                "R_arm_link23",
                "R_arm_link24",
                "R_arm_link25",
                "R_arm_link26",
            ],
        ],
        finger_keys=[["left_arm_tcp_link"], ["right_arm_tcp_link"]],
    )
    agilex_kinematics_config = dict(
        urdf=urdf,
        arm_joint_id=[list(range(9, 15)), list(range(17, 23))],
        arm_link_keys=[
            [
                "left/link1",
                "left/link2",
                "left/link3",
                "left/link4",
                "left/link5",
                "left/link6",
            ],
            [
                "right/link1",
                "right/link2",
                "right/link3",
                "right/link4",
                "right/link5",
                "right/link6",
            ],
        ],
        finger_keys=[
            [
                "left/link7",
            ],
            [
                "right/link7",
            ],
        ],
    )
    genie1_kinematics_config = dict(
        urdf=urdf,
        arm_joint_id=[list(range(4, 11)), list(range(19, 26))],
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
            [
                "gripper_l_center_link",
            ],
            [
                "gripper_r_center_link",
            ],
        ],
    )

    if robot_type == "ARX Lift-2":
        kinematics_config = arx_kinematics_config
        scale_shift = [
            [2.620290994644165, 0.5263252258300781],
            [1.9029523134231567, 1.7795947790145874],
            [2.681088447570801, 0.5804547071456909],
            [1.751371145248413, 0.06441664695739746],
            [2.6713993549346924, -0.9997473359107971],
            [2.1706695556640625, 0.0764995813369751],
            [0.05000000074505806, 0.05000000074505806],
            [2.7875142097473145, 0.3540869951248169],
            [1.8859370946884155, 1.7831171751022339],
            [2.947354316711426, 0.3703957796096802],
            [1.7315316200256348, -0.028902530670166016],
            [1.6735796928405762, -0.0009430050849914551],
            [2.9035515785217285, -0.2364501953125],
            [0.05000000074505806, 0.05000000074505806],
        ]
    elif robot_type == "AgileX Split Aloha":
        kinematics_config = agilex_kinematics_config
        scale_shift = [
            [1.478021398, 0.10237011399999996],
            [1.453678296, 1.4043815520000003],
            [1.553963852, -1.5014923],
            [1.86969153, -0.0010728060000000372],
            [1.3381379620000002, -0.012585846000000012],
            [3.086157592, -0.06803160000000008],
            [0.03857, 0.036329999999999994],
            [1.478021398, 0.10237011399999996],
            [1.453678296, 1.4043815520000003],
            [1.553963852, -1.5014923],
            [1.86969153, -0.0010728060000000372],
            [1.3381379620000002, -0.012585846000000012],
            [3.086157592, -0.06803160000000008],
            [0.03857, 0.036329999999999994],
        ]
    elif robot_type == "Genie-1":
        kinematics_config = genie1_kinematics_config
        scale_shift = [
            [1.478021398, 0.10237011399999996],
            [1.453678296, 1.4043815520000003],
            [1.553963852, -1.5014923],
            [1.86969153, -0.0010728060000000372],
            [1.3381379620000002, -0.012585846000000012],
            [3.086157592, -0.06803160000000008],
            [3.086157592, -0.06803160000000008],
            [0.03857, 0.036329999999999994],
            [1.478021398, 0.10237011399999996],
            [1.453678296, 1.4043815520000003],
            [1.553963852, -1.5014923],
            [1.86969153, -0.0010728060000000372],
            [1.3381379620000002, -0.012585846000000012],
            [3.086157592, -0.06803160000000008],
            [3.086157592, -0.06803160000000008],
            [0.03857, 0.036329999999999994],
        ]
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    kinematics = dict(type=MultiArmKinematics, **kinematics_config)

    if do_calib_to_ext:
        calib_to_ext = dict(
            type=CalibrationToExtrinsic,
            calibration=calibration,
            cam_ee_joint_indices=dict(left=5, right=12),
            **kinematics_config,
        )
    else:
        calib_to_ext = dict(type=IdentityTransform)

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
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
            ego_to_cam,
            projection_mat,
            scale_shift,
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
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
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
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


def _build_lmdb_dataset(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode,
    lazy_init=True,
):
    from robo_orchard_lab.dataset.interna1 import (
        InternA1LmdbDataset,
    )

    dataset_lmdb_config = get_dataset_lmdb_config()
    data_config = dataset_lmdb_config[setting_type]
    transforms = build_lmdb_transforms(
        config,
        mode,
        urdf=data_config["urdf"],
        robot_type=data_config["robot_type"],
        calibration=data_config.get("calibration"),
        depth_restore=config.get("depth_restore", False),
        do_calib_to_ext=not data_config.get("load_extrinsic", False),
    )
    if isinstance(data_paths, list):
        data_paths = sorted(data_paths)
    return InternA1LmdbDataset(
        paths=data_paths,
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        cam_names=data_config["cam_names"],
        task_names=data_config.get("task_names"),
        load_extrinsic=data_config.get("load_extrinsic", False),
        load_calibration=data_config.get("load_calibration", False),
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        reset_step=200,
    )


@train_dataset_register(DATA_TYPE)
def build_lmdb_datasets(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode="training",
    lazy_init=True,
):
    return _build_lmdb_dataset(
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

    dataset_lmdb_config = get_dataset_lmdb_config()
    data_config = dataset_lmdb_config[setting_type]
    transforms = build_lmdb_transforms(
        config,
        mode="deploy",
        urdf=data_config["urdf"],
        robot_type=data_config["robot_type"],
        calibration=data_config.get("calibration"),
        depth_restore=config.get("depth_restore", False),
        do_calib_to_ext=True,
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            cam_names=data_config["cam_names"],
            transforms=transforms,
        )
    )


@processor_register(DATA_TYPE)
def build_processors(
    config,
    dataset_name,
    setting_type,
):
    return _build_processor(config, setting_type=setting_type)


# build_datasets = build_arrow_datasets
build_datasets = build_lmdb_datasets
