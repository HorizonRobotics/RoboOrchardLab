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

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

all_calibrations = dict(
    challenge=dict(
        front={
            "position": [
                -0.010783568385050412,
                -0.2559182030838615,
                0.5173197227547938,
            ],
            "orientation": [
                -0.6344593881273598,
                0.6670669773214551,
                -0.2848079166270871,
                0.2671467447131103,
            ],
        },
        middle={
            "position": [
                -0.010783568385050412,
                -0.2559182030838615,
                0.5173197227547938,
            ],
            "orientation": [
                -0.6344593881273598,
                0.6670669773214551,
                -0.2848079166270871,
                0.2671467447131103,
            ],
        },
        left={
            "position": [-0.0693628, 0.04614798, 0.02938585],
            "orientation": [
                -0.13265687,
                0.13223542,
                -0.6930087,
                0.69615791,
            ],
        },
        right={
            "position": [-0.0693628, 0.04614798, 0.02938585],
            "orientation": [
                -0.13265687,
                0.13223542,
                -0.6930087,
                0.69615791,
            ],
        },
    ),
    horizon_beijing=dict(
        mid={
            "position": [
                -0.010783568385050412,
                -0.2559182030838615,
                0.5173197227547938,
            ],
            "orientation": [
                -0.6344593881273598,
                0.6670669773214551,
                -0.2848079166270871,
                0.2671467447131103,
            ],
        },
        middle={
            "position": [
                -0.010783568385050412,
                -0.2559182030838615,
                0.5173197227547938,
            ],
            "orientation": [
                -0.6344593881273598,
                0.6670669773214551,
                -0.2848079166270871,
                0.2671467447131103,
            ],
        },
        left={
            "position": [
                -0.06867924193484086,
                -0.0005945544447201671,
                0.03843362824412718,
            ],
            "orientation": [
                -0.14277810176817451,
                0.1236499359266293,
                -0.6680764786273947,
                0.7197214222917346,
            ],
        },
        right={
            "position": [
                -0.07333788908459828,
                0.00991803705544634,
                0.03390080995535155,
            ],
            "orientation": [
                0.1296176811682453,
                -0.12171535345636147,
                0.717362436615576,
                -0.673628802824318,
            ],
        },
    ),
    horizon_shanghai=dict(
        middle={
            "position": [
                0.012228904199106883,
                -0.2551978924072771,
                0.5421855239146341,
            ],
            "orientation": [
                0.6517616496718316,
                -0.6383174678039593,
                0.3024538123832749,
                -0.2761869904398411,
            ],
        },
        front={
            "position": [
                0.012228904199106883,
                -0.2551978924072771,
                0.5421855239146341,
            ],
            "orientation": [
                0.6517616496718316,
                -0.6383174678039593,
                0.3024538123832749,
                -0.2761869904398411,
            ],
        },
        left={
            "position": [
                -0.07772281923517543,
                -0.016177539816471194,
                0.11494848529032653,
            ],
            "orientation": [
                -0.14243538719273263,
                0.12358797697833865,
                -0.5747293120148086,
                0.796319276630073,
            ],
        },
        right={
            "position": [
                -0.07785751928191678,
                -0.011973702698755098,
                0.022057130159911904,
            ],
            "orientation": [
                -0.14710187691485124,
                0.1219132257107957,
                -0.6695328337088485,
                0.717791047443973,
            ],
        },
    ),
)


def get_data_paths(dataset_name):
    from glob import glob

    data_paths = []
    if dataset_name == "challenge":
        patterns = [
            "./data/challenge/倒水*/*",
            "./data/challenge/叠毛巾*/叠毛巾-白黑格纹/*",
            "./data/challenge/叠毛巾*/叠毛巾517mm*",
            "./data/challenge/叠盘子*/*",
            "./data/challenge/叠短裤*/*",
            "./data/challenge/盖笔帽*/*",
        ]
    elif dataset_name == "horizon_beijing":
        patterns = [
            "./data/horizon_beijing/xuewu.lin-empty_cup_place",
            "./data/horizon_beijing/xuewu.lin-collect_bottles",
            "./data/horizon_beijing/xuewu.lin-collect_bottles-20250707-v2",
            "./data/horizon_beijing/xuewu.lin-place_to_slot",
            "./data/horizon_beijing/xuewu.lin-place_to_slot-20250709",
            "./data/horizon_beijing/xuewu.lin-two_fold_towel-20250710",
            "./data/horizon_beijing/xuewu.lin-two_fold_towel-20250712",
            "./data/horizon_beijing/zhixu.zhao-*",
            "./data/horizon_beijing/*-place_objects_to_basket-*",
            "./data/horizon_beijing/*-fold_paper_box-*",
        ]
    elif dataset_name == "agilex":
        patterns = ["./data/agilex_collect/lmdb_dataset*"]
    else:
        raise ValueError
    for pattern in patterns:
        data_paths.extend(glob(pattern))
    data_paths = list(set(data_paths))
    data_paths.sort()
    return data_paths


dataset_config = dict(
    challenge=dict(
        data_paths=lambda: get_data_paths("challenge"),
        calibration=all_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_old.urdf",
        cam_names=["left", "right", "front"],
        load_extrinsic=False,
        depth_restore=True,
        flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "challenge").hex[:4], 16),
    ),
    challenge_finetune=dict(
        data_paths=["./data/challenge/finetune"],
        calibration=all_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "front"],
        load_extrinsic=False,
    ),
    challenge_self_collect=dict(
        data_paths=[
            "./data/challenge/agilex_data_0527_plates_stack",
            "./data/challenge/agilex_data_0525",
        ],
        calibration=all_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
    ),
    horizon_beijing=dict(
        data_paths=lambda: get_data_paths("horizon_beijing"),
        calibration=all_calibrations["horizon_beijing"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
    ),
    # Before re-calibration, Firmware version 1.6.5
    horizon_shanghai_0804=dict(
        data_paths=[
            "./data/horizon_shanghai/agilex_empty_cup_place_2025_08_13",
            "./data/horizon_shanghai/agilex_empty_cup_place_2025_08_19",
            "./data/horizon_shanghai/agilex_place_shoe_2025_08_21",
            "./data/horizon_shanghai/agilex_place_shoe_2025_08_27",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_08_05",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_08_07",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_08_08",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_08_12",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_08_27",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_08_28",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_09_01",
            "./data/horizon_shanghai/agilex_place_to_slot_2025_09_02",
            "./data/horizon_shanghai/agilex_put_bottles_dustbin_2025_08_20",
            "./data/horizon_shanghai/agilex_put_bottles_dustbin_2025_08_21",
            "./data/horizon_shanghai/agilex_stack_blocks_three_2025_08_14",
            "./data/horizon_shanghai/agilex_stack_blocks_three_2025_08_15",
            "./data/horizon_shanghai/agilex_stack_blocks_three_2025_08_18",
            "./data/horizon_shanghai/agilex_stack_blocks_three_2025_08_26",
            "./data/horizon_shanghai/agilex_stack_blocks_three_2025_08_27",
            "./data/horizon_shanghai/agilex_stack_bowls_three_2025_08_15",
            "./data/horizon_shanghai/agilex_stack_bowls_three_2025_08_18",
            "./data/horizon_shanghai/agilex_stack_bowls_three_2025_08_19",
            "./data/horizon_shanghai/agilex_stack_bowls_three_2025_08_20",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_04",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_06",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_07",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_08",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_11",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_13",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_14",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_22",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_25",
            "./data/horizon_shanghai/agilex_two_fold_towel_2025_08_26",
        ],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "middle"],
        task_names=[
            "empty_cup_place",
            "place_shoe",
            "place_to_slot",
            "put_bottles_dustbin",
            "stack_blocks_three",
            "stack_bowls_three",
            "two_fold_towel",
        ],
        load_extrinsic=True,
    ),
    # After re-calibration, Firmware version 1.8.0
    horizon_shanghai_0909=dict(
        data_paths=[
            "./data/horizon_shanghai/lmdb_dataset_empty_cup_place_2025_09_09",
            "./data/horizon_shanghai/lmdb_dataset_place_shoe_2025_09_11",
            "./data/horizon_shanghai/lmdb_dataset_place_to_slot_2025_09_15",
            "./data/horizon_shanghai/lmdb_dataset_place_to_slot_2025_09_16",
            "./data/horizon_shanghai/lmdb_dataset_place_to_slot_2025_09_17",
            "./data/horizon_shanghai/lmdb_dataset_place_to_slot_2025_09_18",
            "./data/horizon_shanghai/lmdb_dataset_place_to_slot_2025_09_22",
            "./data/horizon_shanghai/lmdb_dataset_put_bottles_dustbin_2025_09_11",
            "./data/horizon_shanghai/lmdb_dataset_stack_block_two_2025_09_17",
            "./data/horizon_shanghai/lmdb_dataset_stack_blocks_three_2025_09_10",
            "./data/horizon_shanghai/lmdb_dataset_stack_bowls_three_2025_09_09",
            "./data/horizon_shanghai/lmdb_dataset_stack_bowls_three_2025_09_10",
            "./data/horizon_shanghai/lmdb_dataset_two_fold_towel_2025_09_12",
            "./data/horizon_shanghai/lmdb_dataset_two_fold_towel_2025_09_23",
        ],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "middle"],
        task_names=[
            "empty_cup_place",
            "place_shoe",
            "place_to_slot",
            "put_bottles_dustbin",
            "stack_blocks_three",
            "stack_bowls_three",
            "two_fold_towel",
        ],
        load_extrinsic=True,
    ),
    # Agilex External Dataset
    agilex=dict(
        data_paths=lambda: get_data_paths("agilex"),
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "mid"],
        task_names=[
            "fold_towel",
            "pour_water",
            "plug_charger",
            "hand_out",
            "make_coffee",
            "buss_table",
            "spell_yes",
            "pick_larger_value",
            "sort_toy_by_color",
            "ziploc_slide",
            "twist_off_the_cap",
            "unplug_the_charging_cable",
            "wash_pan",
            "wipe_wine",
            "move_chair",
        ],
        calibration=all_calibrations["horizon_beijing"],
        load_extrinsic=True,
    ),
)


def build_transforms(
    config,
    mode,
    urdf,
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

    if depth_restore:
        depth_restoration = dict(type=DepthRestoration)
    else:
        depth_restoration = dict(type=IdentityTransform)

    t_base2ego = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]]
    ).tolist()  # noqa: N806
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

    kinematics_config = dict(
        urdf=urdf,
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
    )
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

    scale_shift = dict(
        type=AddScaleShift,
        scale_shift=[
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
        ],
    )
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
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            random_crop_padding,
            resize,
            to_tensor,
            calib_to_ext,
            extrinsic_noise,
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


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    from robo_orchard_lab.dataset.horizon_manipulation import (
        HorizonManipulationLmdbDataset,
    )
    from robo_orchard_lab.dataset.lmdb.instruction_reader import (
        InstructionReader,
    )

    datasets = {}
    for dataset_name, data_config in dataset_config.items():
        if dataset_name not in dataset_names:
            continue
        transforms = build_transforms(
            config,
            mode,
            urdf=data_config["urdf"],
            calibration=data_config.get("calibration"),
            depth_restore=config.get("depth_restore", False),
            do_calib_to_ext=not data_config.get("load_extrinsic", False),
        )
        instruction_reader = dict(
            type=InstructionReader,
            lmdb_path="./data/instructions/subtasks_agibot_rh20t_agilex_20250714/",
            instruction_path="./data/instructions/task2instruction_0928.json",
        )
        data_paths = data_config["data_paths"]
        if callable(data_paths):
            data_paths = data_paths()
        dataset = HorizonManipulationLmdbDataset(
            paths=data_paths,
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=dataset_name,
            cam_names=data_config["cam_names"],
            task_names=data_config.get("task_names"),
            load_extrinsic=data_config.get("load_extrinsic", False),
            load_calibration=data_config.get("load_calibration", False),
            instruction_reader=instruction_reader,
            flag=data_config.get(
                "flag",
                int(uuid.uuid5(uuid.NAMESPACE_DNS, "agilex").hex[:4], 16),
            ),
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
        )
        datasets[dataset_name] = dataset

    return datasets


@processor_register()
def build_processors(config, dataset_names):
    from robo_orchard_lab.models.sem_modules import (
        SEMProcessor,
        SEMProcessorCfg,
    )

    processors = {}
    for dataset_name, data_config in dataset_config.items():
        if dataset_name not in dataset_names:
            continue

        transforms = build_transforms(
            config,
            mode="deploy",
            urdf=data_config["urdf"],
            calibration=data_config.get("calibration"),
            depth_restore=config.get("depth_restore", False),
            do_calib_to_ext=True,
        )
        processor = SEMProcessor(
            SEMProcessorCfg(
                load_image=True,
                load_depth=config["with_depth"],
                valid_action_step=None,
                cam_names=data_config["cam_names"],
                transforms=transforms,
            )
        )
        processors[dataset_name] = processor
    return processors
