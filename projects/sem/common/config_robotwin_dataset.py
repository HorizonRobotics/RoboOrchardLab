def build_transforms(
    config, mode, urdf="./urdf/arx5/arx5_description_isaac.urdf"
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

    t_base2world = np.array(
        [
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    ee_state_loss_weights = [1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * 6
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * 6
            + [ee_state_loss_weights]
        ],
    )

    if mode == "training":
        add_data_relative_items = AddItems(
            T_base2world=t_base2world,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
        )
    else:
        add_data_relative_items = AddItems(
            T_base2world=t_base2world,
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

    kinematics = DualArmKinematics(urdf=urdf)

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
    if "robotwin1.0" in dataset_names:
        data_paths = [
            "./data/robotwin1.0",
        ]
        transforms = build_transforms(config, mode)
        dataset = RoboTwinLmdbDataset(
            paths=data_paths,
            task_names=None,
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name="robotwin1.0",
        )
        datasets.append(dataset)
    return datasets
