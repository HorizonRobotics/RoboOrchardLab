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

DATA_TYPE = "agilex"

default_calibrations = dict(
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
    horizon_piper_435_low_beijing=dict(
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
    horizon_piper_435_low_shanghai=dict(
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
    horizon_piper_435_high=dict(
        left={
            "position": [
                -0.07187119209436892,
                -0.0038742124545078963,
                0.03848311081942217,
            ],
            "orientation": [
                -0.14929331497130632,
                0.1386757370673157,
                -0.6856238984540931,
                0.6988565059597904,
            ],
        },
        right={
            "position": [
                -0.07020379397980514,
                0.002208931101560816,
                0.03864419615086635,
            ],
            "orientation": [
                -0.12810968024634153,
                0.12425625721361307,
                -0.6940195330118537,
                0.6974848960145682,
            ],
        },
        middle={
            "position": [
                -0.03007155725467936,
                -0.2730860893744077,
                0.7733328288568009,
            ],
            "orientation": [
                -0.6690798281518483,
                0.6919193985821155,
                -0.19544146988141584,
                0.1881019970033503,
            ],
        },
    ),
    horizon_piper_x_435=dict(
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
                -0.0066753085430642164,
                -0.07067909189160133,
                0.04071981595986668,
            ],
            "orientation": [
                -0.16576259418303393,
                -0.001324287215046063,
                0.009535575604591298,
                0.9861186954068825,
            ],
        },
        right={
            "position": [
                -0.014657614694640091,
                -0.08231392542130013,
                0.05495365864846876,
            ],
            "orientation": [
                -0.1878112655573414,
                0.01474422824253986,
                -0.020030077316378438,
                0.9818901834044282,
            ],
        },
    ),
    horizon_piper_x_405_455=dict(
        left={
            "position": [
                -0.0096489170911759,
                -0.08009372951791657,
                0.04279548930003773,
            ],
            "orientation": [
                -0.17191546428297078,
                -0.0013898166917794218,
                -0.008866866103733552,
                0.9850708199086157,
            ],
        },
        right={
            "position": [
                -0.0076363572782560665,
                -0.07947460457157493,
                0.043216980311924016,
            ],
            "orientation": [
                -0.1764401074959075,
                0.0001554395920107693,
                -0.0032741894713766953,
                0.9843059199195499,
            ],
        },
        middle={
            "position": [
                -0.007383059883329407,
                -0.31715773707478656,
                0.6036358425132415,
            ],
            "orientation": [
                -0.6620275905116694,
                0.6913463014325211,
                -0.2114074909007769,
                0.19765281097906265,
            ],
        },
    ),
)


dataset_config = dict(
    challenge=dict(
        default_calibration=default_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_old.urdf",
        cam_names=["left", "right", "front"],
        load_extrinsic=False,
        depth_restore=True,
        flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "challenge").hex[:4], 16),
    ),
    challenge_finetune=dict(
        default_calibration=default_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "front"],
        load_extrinsic=False,
    ),
    challenge_self_collect=dict(
        default_calibration=default_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
    ),
    horizon_piper_435_low_beijing=dict(
        default_calibration=default_calibrations["horizon_piper_435_low_beijing"],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
    ),
    horizon_piper_435_low_shanghai=dict(
        default_calibration=default_calibrations["horizon_piper_435_low_shanghai"],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
    ),
    horizon_piper_435_high=dict(
        default_calibration=default_calibrations["horizon_piper_435_high"],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
    ),
    horizon_piper_x_435=dict(
        default_calibration=default_calibrations["horizon_piper_x_435"],
        urdf="./urdf/piper_x_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
        flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "piper_x").hex[:4], 16),
    ),
    horizon_piper_x_405_455=dict(
        default_calibration=default_calibrations["horizon_piper_x_405_455"],
        urdf="./urdf/piper_x_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        load_extrinsic=True,
        flag=int(uuid.uuid5(uuid.NAMESPACE_DNS, "piper_x").hex[:4], 16),
    ),
    # Agilex External Dataset
    agilex=dict(
        urdf="./urdf/piper_description_dualarm.urdf",
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
        default_calibration=default_calibrations["horizon_piper_435_low_beijing"],
        load_extrinsic=True,
    ),
)

def build_transforms(
    config,
    mode,
    urdf,
    default_calibration=None,
    depth_restore=False,
    do_calib_to_ext=False,
    truncated_subtask=False,
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
        TruncatedTrajectoryBySubtask,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.transforms import ValueSampling

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

    value_sampling = (
        dict(
            type=ValueSampling,
            norm_mode=config["value_norm_mode"],
            task_max_step=None,
        )
        if config.get("value_model_training", False)
        else None
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
            calibration=default_calibration,
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
        if truncated_subtask:
            truncated_subtask = TruncatedTrajectoryBySubtask()
        else:
            truncated_subtask = IdentityTransform()
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
                "value",
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
            truncated_subtask,
            depth_restoration,
            add_data_relative_items,
            value_sampling,
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
                "value",
            ],
        )
        transforms = [
            depth_restoration,
            add_data_relative_items,
            value_sampling,
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


@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_dataset(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode,
    instruction_paths=None,
    lazy_init=True,
    truncated_subtask=False,
):
    from robo_orchard_lab.dataset.horizon_manipulation import (
        HorizonManipulationLmdbDataset,
    )
    from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
        InstructionReader,
    )

    data_config = dataset_config[setting_type]
    transforms = build_transforms(
        config,
        mode,
        urdf=data_config["urdf"],
        default_calibration=data_config.get("default_calibration"),
        depth_restore=config.get("depth_restore", False),
        do_calib_to_ext=not data_config.get("load_extrinsic", False),
        truncated_subtask=truncated_subtask,
    )
    if instruction_paths is not None:
        instruction_reader = InstructionReader(paths=instruction_paths)
    else:
        instruction_reader = None
    if callable(data_paths):
        data_paths = data_paths()
    return HorizonManipulationLmdbDataset(
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
        reset_step=500,
    )


def _build_processor(config, setting_type):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    data_config = dataset_config[setting_type]
    transforms = build_transforms(
        config,
        mode="deploy",
        urdf=data_config["urdf"],
        default_calibration=data_config.get("default_calibration"),
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
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    data_config = dataset_config[setting_type]
    transforms = build_transforms(
        config,
        mode="deploy",
        urdf=data_config["urdf"],
        default_calibration=data_config.get("default_calibration"),
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
