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

dataset_config = dict(
    challenge=dict(
        data_paths=[
            "./data/challenge/倒水-AH202502130001/倒水517mm-灰白",
            "./data/challenge/倒水-AH202502130001/倒水517mm-绿白",
            "./data/challenge/倒水-AH202502130001/倒水517mm",
            "./data/challenge/叠毛巾-AH202502130002/叠毛巾-白黑格纹/叠毛巾-517-3-白黑格纹",
            "./data/challenge/叠毛巾-AH202502130002/叠毛巾-白黑格纹/叠毛巾-517-3-白黑格纹-2",
            "./data/challenge/叠毛巾-AH202502130002/叠毛巾-白黑格纹/叠毛巾补数据",
            "./data/challenge/叠毛巾-AH202502130002/叠毛巾517mm-灰白",
            "./data/challenge/叠毛巾-AH202502130002/叠毛巾517mm-白背景",  # 7
            "./data/challenge/叠毛巾-AH202502130002/叠毛巾517mm-绿白",
            "./data/challenge/叠盘子-AH202501130003/叠盘子-517-3-绿白格纹",
            "./data/challenge/叠盘子-AH202501130003/叠盘子517mm-1",
            "./data/challenge/叠盘子-AH202501130003/叠盘子补采集-灰白",
            "./data/challenge/叠盘子-AH202501130003/叠盘子补采集-黑色桌面",
            "./data/challenge/叠短裤-AH202503060005/叠短裤517mm-灰白",  # 13
            "./data/challenge/叠短裤-AH202503060005/叠短裤517mm-绿白",
            "./data/challenge/叠短裤-AH202503060005/叠短裤517mm-青白",
            "./data/challenge/盖笔帽-AH202501250001/盖笔帽-517-3-绿白格纹",
            "./data/challenge/盖笔帽-AH202501250001/盖笔帽-517-3-灰白格纹",
            "./data/challenge/盖笔帽-AH202501250001/盖笔帽-517-3-白黑格纹",
        ],
        calibration=all_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_old.urdf",
        cam_names=["left", "front", "right"],
        load_extrinsic=False,
        depth_restore=True,
    ),
    challenge_finetune=dict(
        data_paths=[
            "./data/challenge/finetune",
        ],
        calibration=all_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "front", "right"],
        load_extrinsic=False,
    ),
    challenge_self_collect=dict(
        data_paths=[
            "./data/challenge/agilex_data_0527_plates_stack",
            "./data/challenge/agilex_data_0525",
        ],
        calibration=all_calibrations["challenge"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "middle", "right"],
        load_extrinsic=True,
    ),
    horizon_beijing=dict(
        data_paths=[
            "./data/horizon_beijing/agilex_data_0424_blocks_stack_hard",
            "./data/horizon_beijing/agilex_data_0425_shoe_place",
            "./data/horizon_beijing/agilex_data_0425_blocks_stack_easy",
            "./data/horizon_beijing/agilex_data_0428_blocks_stack_easy",
            "./data/horizon_beijing/agilex_data_0428_blocks_stack_hard",
            "./data/horizon_beijing/agilex_data_0429_diverse_bottles_pick",
            "./data/horizon_beijing/agilex_data_0429-22_diverse_bottles_pick",
            "./data/horizon_beijing/xuewu.lin-empty_cup_place",
            "./data/horizon_beijing/xuewu.lin-collect_bottles",
            "./data/horizon_beijing/xuewu.lin-collect_bottles-20250707-v2",
            "./data/horizon_beijing/xuewu.lin-place_to_slot",
            "./data/horizon_beijing/xuewu.lin-place_to_slot-20250709",
            "./data/horizon_beijing/xuewu.lin-two_fold_towel-20250710",
            "./data/horizon_beijing/xuewu.lin-two_fold_towel-20250712",
        ],
        calibration=all_calibrations["horizon_beijing"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "middle", "right"],
        load_extrinsic=True,
    ),
    horizon_shanghai=dict(
        data_paths=[
            "./data/horizon_shanghai/agilex_two_fold_towel_0623",
            "./data/horizon_shanghai/agilex_place_to_slot_0624",
            "./data/horizon_shanghai/agilex_place_to_slot_0626",
            "./data/horizon_shanghai/agilex_collect_bottles_01_0708",
            "./data/horizon_shanghai/agilex_collect_bottles_02_0708",
            "./data/horizon_shanghai/agilex_collect_bottles_03_0708",
            "./data/horizon_shanghai/agilex_collect_bottles_0711",
            "./data/horizon_shanghai/agilex_empty_cup_place_01_0710",
            "./data/horizon_shanghai/agilex_empty_cup_place_02_0710",
        ],
        calibration=all_calibrations["horizon_shanghai"],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "middle", "right"],
        load_extrinsic=True,
    ),
    agilex=dict(
        data_paths=[
            # f"./data/agilex_collect/agilex_data_0624_shard_{i}"
            # for i in range(12)
            f"data/agilex_collect/agitex_piper_0522_shard_{i}"
            for i in range(13)
        ]
        + [
            f"data/agilex_collect/agilex_old_urdf/old_urdf_0523_shard_{i}"
            for i in range(4)
        ],
        load_extrinsic=False,
        # load_calibration=True,
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "mid", "right"],
        calibration=all_calibrations["horizon_beijing"],
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
            # exclude three mobile manipulation tasks
            # "wash_pan",
            # "wipe_wine",
            # "move_chair",
        ],
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
    import torch

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        CalibrationToExtrinsic,
        ConvertDataType,
        DepthRestoration,
        GetProjectionMat,
        IdentityTransform,
        ItemSelection,
        JointStateNoise,
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

    t_base2ego = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0.3], [0, 0, 1, 0], [0, 0, 0, 1]]
    ).tolist()  # noqa: N806
    t_base2world = np.eye(4).tolist()  # noqa: N806

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
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
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
        )
        transforms = [
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
            projection_mat,
            scale_shift,
            # joint_state_noise,
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
            ],
        )
        transforms = [
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
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
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            calib_to_ext,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


def build_datasets(config, dataset_names, mode, lazy_init=True):
    from robo_orchard_lab.dataset.horizon_manipulation import (
        HorizonManipulationLmdbDataset,
    )

    datasets = []
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
        dataset = HorizonManipulationLmdbDataset(
            paths=data_config["data_paths"],
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=dataset_name,
            cam_names=data_config["cam_names"],
            task_names=data_config.get("task_names"),
            load_extrinsic=data_config.get("load_extrinsic", False),
            load_calibration=data_config.get("load_calibration", False),
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
