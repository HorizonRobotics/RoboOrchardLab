dataset_config = dict(
    robotwin1_0=dict(
        kinematics_config=dict(
            urdf="./urdf/arx5/arx5_description_isaac.urdf",
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        paths=[
            "./data/robotwin1.0",
        ],
    ),
    robotwin2_0_piper=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin2_dual_arm_piper.urdf",
            left_arm_link_keys=[
                "left_link1",
                "left_link2",
                "left_link3",
                "left_link4",
                "left_link5",
                "left_link6",
            ],
            left_finger_keys=["left_link7"],
            right_arm_link_keys=[
                "right_link1",
                "right_link2",
                "right_link3",
                "right_link4",
                "right_link5",
                "right_link6",
            ],
            right_finger_keys=["right_link7"],
            left_arm_joint_id=list(range(6)),
            right_arm_joint_id=list(range(8, 14)),
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.45],
            [0, 0, 1, 0.75],
            [0, 0, 0, 1],
        ],
        paths=[
            "./data/robotwin2.0/aloha_piper_27tasks_clean_200",
            "./data/robotwin2.0/aloha_piper_27tasks_noise_300",
        ],
    ),
)


def build_transforms(
    config, mode, kinematics_config, t_base2world
):
    import numpy as np
    import torch

    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        DualArmKinematics,
        GetProjectionMat,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )

    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    ee_state_loss_weights = [1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]
    loss_weights = np.array([
        [joint_state_loss_weights] * 6
        + [ee_state_loss_weights]
        + [joint_state_loss_weights] * 6
        + [ee_state_loss_weights]
    ])

    if mode == "training":
        add_data_relative_items = AddItems(
            T_base2world=np.array(t_base2world),
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
        )
    else:
        add_data_relative_items = AddItems(
            T_base2world=np.array(t_base2world),
        )

    state_sampling = SimpleStateSampling(
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = Resize(
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = ImageChannelFlip([2, 1, 0])
    to_tensor = ToTensor()
    projection_mat = GetProjectionMat(target_coordinate="base")
    convert_dtype = ConvertDataType(
        convert_map=dict(
            imgs=torch.float32,
            depths=torch.float32,
            image_wh=torch.float32,
            projection_mat=torch.float32,
            embodiedment_mat=torch.float32,
        )
    )

    kinematics = DualArmKinematics(**kinematics_config)

    scale_shift = AddScaleShift(
        scale_shift=[
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.0325, 0.0125],
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.0325, 0.0125],
        ]
    )
    if mode == "training":
        item_selection = ItemSelection(
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
            ]
        )
        joint_state_noise = JointStateNoise(
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
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            projection_mat,
            scale_shift,
            joint_state_noise,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "validation":
        item_selection = ItemSelection(
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
            ]
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        item_selection = ItemSelection(
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
            ]
        )
        unsqueeze_batch = UnsqueezeBatch()
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


def build_datasets(config, dataset_names, mode, lazy_init=True):
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    datasets = []
    for dataset_name in dataset_config.keys():
        if "robotwin" not in dataset_names and dataset_name not in dataset_names:
            continue
        transforms = build_transforms(
            config, mode, dataset_config[dataset_name]['kinematics_config'],
            dataset_config[dataset_name]['T_base2world'],
        )
        dataset = RoboTwinLmdbDataset(
            paths=dataset_config[dataset_name]['paths'],
            task_names=None,
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=dataset_name,
        )
        datasets.append(dataset)
    return datasets
