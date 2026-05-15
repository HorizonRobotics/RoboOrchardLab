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


import json

import numpy as np
from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

dataset_config = dict(
    ur5=dict(
        kinematics=dict(
            urdf="./urdf/ur5_wsg_gripper.urdf",
            arm_joint_id=[list(range(6))],
            arm_link_keys=[
                [
                    "shoulder_link",
                    "upper_arm_link",
                    "forearm_link",
                    "wrist_1_link",
                    "wrist_2_link",
                    "wrist_3_link",
                ],
            ],
            finger_keys=[["fingertip_tip_center_link"]],
        ),
        scale_shift=[
            [0.844146, 0.063639],  # joint0
            [0.881836, -2.071944],  # joint1
            [1.458221, -0.767419],  # joint2
            [1.631188, -1.311326],  # joint3
            [0.401934, 1.577625],  # joint4
            [3.112092, 0.398812],  # joint5
            [0.042500, 0.042500],  # gripper
        ],
        data_paths=[
            "./data/table30v2/lmdb/ur5/arrange_fruits",
            "./data/table30v2/lmdb/ur5/item_classification",
            "./data/table30v2/lmdb/ur5/shred_paper",
        ],
        cam_names=["cam_arm", "cam_global"],
        ee_link=[["tool0"]],
        cam_ee_joint_indices=dict(cam_arm=6),
    ),
    arx5=dict(
        kinematics=dict(
            urdf="./urdf/X5A_recorded_ee.urdf",
            arm_joint_id=[list(range(6))],
            arm_link_keys=[
                [
                    "link1",
                    "link2",
                    "link3",
                    "link4",
                    "link5",
                    "link6",
                ],
            ],
            finger_keys=[["recorded_ee_link"]],
        ),
        scale_shift=[
            [1.803617, 0.073052],  # joint0
            [1.427100, 1.426528],  # joint1
            [1.461624, 1.456092],  # joint2
            [1.585222, -0.022507],  # joint3
            [1.453422, 0.020791],  # joint4
            [1.998360, -0.000381],  # joint5
            [0.043551, 0.043694],  # gripper
        ],
        data_paths=[
            "./data/table30v2/lmdb/arx5/arrange_flowers",
            "./data/table30v2/lmdb/arx5/hang_the_cup",
            "./data/table30v2/lmdb/arx5/pick_out_the_green_blocks",
            # "./data/table30v2/lmdb/arx5/press_the_button",
            "./data/table30v2/lmdb/arx5/turn_on_the_light_switch",
            "./data/table30v2/lmdb/arx5/water_the_flowers",
            "./data/table30v2/lmdb/arx5/wipe_the_table",
        ],
        cam_names=["cam_arm", "cam_side", "cam_global"],
        cam_ee_joint_indices=dict(cam_arm=6),
    ),
    aloha=dict(
        kinematics=dict(
            urdf="./urdf/piper_description_dualarm.urdf",
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            arm_link_keys=[
                [
                    "left_link1",
                    "left_link2",
                    "left_link3",
                    "left_link4",
                    "left_link5",
                    "left_link6",
                ],
                [
                    "right_link1",
                    "right_link2",
                    "right_link3",
                    "right_link4",
                    "right_link5",
                    "right_link6",
                ],
            ],
            finger_keys=[["left_link7"], ["right_link7"]],
        ),
        scale_shift=[
            [1.469177, -0.128763],  # left_joint0
            [1.418223, 1.418223],  # left_joint1
            [1.444730, -1.444730],  # left_joint2
            [1.744400, 0.000000],  # left_joint3
            [1.221080, 0.000000],  # left_joint4
            [2.993085, -0.004544],  # left_joint5
            [0.055550, 0.052850],  # left_gripper
            [1.532848, 0.300970],  # right_joint0
            [1.432911, 1.432911],  # right_joint1
            [1.482740, -1.482740],  # right_joint2
            [1.744400, 0.000000],  # right_joint3
            [1.221080, 0.000000],  # right_joint4
            [2.990137, 0.013127],  # right_joint5
            [0.054850, 0.052250],  # right_gripper
        ],
        data_paths=[
            "./data/table30v2/lmdb/aloha/put_the_books_back",
            "./data/table30v2/lmdb/aloha/stamp_positioning",
            "./data/table30v2/lmdb/aloha/wipe_the_blackboard",
            "./data/table30v2/lmdb/aloha/scoop_with_a_small_spoon",
            # hard tasks
            "./data/table30v2/lmdb/aloha/lint_roller_remove_dirt",
            "./data/table30v2/lmdb/aloha/pack_the_items",
            "./data/table30v2/lmdb/aloha/pack_the_toothbrush_holder",
            "./data/table30v2/lmdb/aloha/paint_jam",
            "./data/table30v2/lmdb/aloha/put_the_pencil_case_into_the_schoolbag",
            "./data/table30v2/lmdb/aloha/wrap_with_a_soft_cloth",
        ],
        cam_ee_joint_indices=dict(cam_left_wrist=5, cam_right_wrist=12),
        cam_names=["cam_left_wrist", "cam_right_wrist", "cam_high"],
    ),
    dos_w1=dict(
        kinematics=dict(
            urdf="./urdf/dos-w1.urdf",
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            arm_link_keys=[
                [
                    "left_link1",
                    "left_link2",
                    "left_link3",
                    "left_link4",
                    "left_link5",
                    "left_link6",
                ],
                [
                    "right_link1",
                    "right_link2",
                    "right_link3",
                    "right_link4",
                    "right_link5",
                    "right_link6",
                ],
            ],
            finger_keys=[["left_end_link"], ["right_end_link"]],
        ),
        scale_shift=[
            [1.489090, -0.648508],  # left_joint0
            [1.430533, -1.251049],  # left_joint1
            [1.618982, 1.529145],  # left_joint2
            [3.003739, 0.008202],  # left_joint3
            [1.779774, 0.001144],  # left_joint4
            [3.012131, -0.011253],  # left_joint5
            [0.035934, 0.035448],  # left_gripper
            [1.523041, 0.535592],  # right_joint0
            [1.367018, -1.187152],  # right_joint1
            [1.621843, 1.530480],  # right_joint2
            [3.006600, 0.004959],  # right_joint3
            [1.794461, 0.013924],  # right_joint4
            [3.013848, 0.001907],  # right_joint5
            [0.036022, 0.035387],  # right_gripper
        ],
        data_paths=[
            "./data/table30v2/lmdb/dos_w1/fold_the_clothes",
            "./data/table30v2/lmdb/dos_w1/stack_bowls",
            "./data/table30v2/lmdb/dos_w1/hold_the_tray_with_both_hands",
            "./data/table30v2/lmdb/dos_w1/place_objects_into_desk_drawer",
            "./data/table30v2/lmdb/dos_w1/tidy_up_the_makeup_table",
            "./data/table30v2/lmdb/dos_w1/put_in_pen_container",
            # hard tasks
            "./data/table30v2/lmdb/dos_w1/put_the_shoes_back",
            "./data/table30v2/lmdb/dos_w1/sweep_the_trash",
            "./data/table30v2/lmdb/dos_w1/tie_a_knot",
            "./data/table30v2/lmdb/dos_w1/untie_the_shoelaces",
        ],
        cam_ee_joint_indices=dict(cam_left_wrist=6, cam_right_wrist=13),
        cam_names=["cam_left_wrist", "cam_right_wrist", "cam_high"],
    ),
)


camera_parameters_file = "./data/table30v2/lmdb/default_camera_parameters.json"


def build_transforms(
    config,
    mode,
    cam_names,
    scale_shift,
    kinematics_config,
    do_calib_to_ext,
    cam_ee_joint_indices,
    ee_link,
    default_calibration,
    default_intrinsic,
):
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        CalibrationToExtrinsic,
        ConvertDataType,
        ExtrinsicNoise,
        GetProjectionMat,
        IdentityTransform,
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

    t_base2ego = np.eye(4).tolist()
    t_base2world = np.eye(4).tolist()

    joint_state_loss_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    num_joint = len(scale_shift)
    if num_joint == 7:  # single arm
        loss_weights = [
            [joint_state_loss_weights] * 6 + [ee_state_loss_weights]
        ]
        joint_mask = [True] * 6 + [False]
    else:
        assert num_joint == 14
        loss_weights = [
            [joint_state_loss_weights] * 6
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * 6
            + [ee_state_loss_weights]
        ]
        joint_mask = [True] * 6 + [False] + [True] * 6 + [False]

    if mode == "training":
        loss_weights = np.array(loss_weights)
        state_loss_weights = loss_weights * 0.2
        fk_loss_weight = loss_weights * 1.8
        state_loss_weights = state_loss_weights.tolist()
        fk_loss_weight = fk_loss_weight.tolist()
        add_data_relative_items = dict(
            type=AddItems,
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
            scale_shift=scale_shift,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
            scale_shift=scale_shift,
        )
    if default_calibration is not None:
        add_data_relative_items["calibration"] = default_calibration
    if default_intrinsic is not None:
        add_data_relative_items["intrinsic"] = default_intrinsic

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        use_master_gripper=False,
    )

    padding_depths = np.zeros([len(cam_names), 2, 2]).tolist()
    add_padding_depths = dict(type=AddItems, depths=padding_depths)

    dst_wh = config.get("dst_wh", (308, 252))
    resize = dict(type=Resize, dst_wh=dst_wh)
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

    if do_calib_to_ext:
        if ee_link is not None:
            kinematics_config = kinematics_config.copy()
            kinematics_config["finger_keys"] = ee_link
        calib_to_ext = dict(
            type=CalibrationToExtrinsic,
            cam_ee_joint_indices=cam_ee_joint_indices,
            **kinematics_config,
        )
    else:
        calib_to_ext = dict(type=IdentityTransform)

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
                "joint_mask",
                "pred_mask",
            ],
        )
        if num_joint == 7:
            joint_state_noise = dict(
                type=JointStateNoise,
                noise_range=[
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.0, 0.0],
                ],
                add_to_pred=True,
            )
        else:
            joint_state_noise = dict(
                type=JointStateNoise,
                noise_range=[
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.0, 0.0],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.02, 0.02],
                    [-0.0, 0.0],
                ],
                add_to_pred=True,
            )
        random_crop_padding = dict(
            type=RandomCropPaddingResize,
            range_w=(-30, 30),
            range_h=(-30, 50),
            range_scale=None,
        )
        extrinsic_noise = dict(
            type=ExtrinsicNoise,
            noise_range=(0.04, 0.04, 0.04, 0.015, 0.015, 0.015),
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            random_crop_padding,
            add_padding_depths,
            resize,
            to_tensor,
            calib_to_ext,
            extrinsic_noise,
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
                "subtask",
                "joint_mask",
            ],
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            add_padding_depths,
            resize,
            to_tensor,
            calib_to_ext,
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
                "remaining_actions",
                "delay_horizon",
                "joint_mask",
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            add_padding_depths,
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


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    from robo_orchard_lab.dataset.horizon_manipulation import (
        HorizonManipulationLmdbDataset,
    )

    use_default_camera_parameters = True
    load_extrinsic = use_default_camera_parameters and False
    load_calibration = not (load_extrinsic or use_default_camera_parameters)
    datasets = {}

    camera_parameters = None

    for dataset_name, data_config in dataset_config.items():
        if (
            "table30v2" not in dataset_names
            and f"table30v2_{dataset_name}" not in dataset_names
        ):
            continue
        if camera_parameters is None and use_default_camera_parameters:
            camera_parameters = json.load(open(camera_parameters_file))
        if use_default_camera_parameters:
            default_calibration = camera_parameters["calibration"][
                dataset_name
            ]
            default_intrinsic = [
                camera_parameters["intrinsic"][dataset_name][cam_name]
                for cam_name in data_config["cam_names"]
            ]
        transforms = build_transforms(
            config,
            mode,
            data_config["cam_names"],
            data_config["scale_shift"],
            data_config["kinematics"],
            do_calib_to_ext=not load_extrinsic,
            cam_ee_joint_indices=data_config["cam_ee_joint_indices"],
            ee_link=data_config.get("ee_link"),
            default_calibration=default_calibration,
            default_intrinsic=default_intrinsic,
        )
        dataset = HorizonManipulationLmdbDataset(
            paths=data_config["data_paths"],
            cam_names=data_config["cam_names"],
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=f"table30v2_{dataset_name}",
            load_depth=False,
            load_calibration=load_calibration,
            load_extrinsic=load_extrinsic,
        )
        datasets[f"table30v2_{dataset_name}"] = dataset
    return datasets


@processor_register()
def build_processors(config, dataset_names):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    processors = {}
    camera_parameters = None
    for dataset_name, data_config in dataset_config.items():
        if (
            "table30v2" not in dataset_names
            and f"table30v2_{dataset_name}" not in dataset_names
        ):
            continue

        if camera_parameters is None:
            camera_parameters = json.load(open(camera_parameters_file))
        default_calibration = camera_parameters["calibration"][dataset_name]
        default_intrinsic = [
            camera_parameters["intrinsic"][dataset_name][cam_name]
            for cam_name in data_config["cam_names"]
        ]
        transforms = build_transforms(
            config,
            "deploy",
            data_config["cam_names"],
            data_config["scale_shift"],
            data_config["kinematics"],
            do_calib_to_ext=True,
            cam_ee_joint_indices=data_config["cam_ee_joint_indices"],
            ee_link=data_config.get("ee_link"),
            default_calibration=default_calibration,
            default_intrinsic=default_intrinsic,
        )
        processor = HoloBrainProcessor(
            HoloBrainProcessorCfg(
                load_image=True,
                load_depth=False,
                valid_action_step=None,
                cam_names=data_config["cam_names"],
                transforms=transforms,
            )
        )
        processors[f"table30v2_{dataset_name}"] = processor
    return processors
