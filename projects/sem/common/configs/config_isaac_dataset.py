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

dataset_config = dict(
    isaac_pick_place=dict(
        data_paths=[
            "/horizon-bucket/robot_lab/users/mengao.zhao/dataset/pick_place_arrow/stack_block_two_seed0-499",  # noqa: E501
            "/horizon-bucket/robot_lab/users/mengao.zhao/dataset/pick_place_arrow/place_mouse_pad_seed0-499",  # noqa: E501
            "/horizon-bucket/robot_lab/users/mengao.zhao/dataset/pick_place_arrow/place_lemon_plate_seed0-499",  # noqa: E501
        ],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=[
            "left_hand_camera",
            "right_hand_camera",
            "camera_ext",
        ],
    )
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
        GetProjectionMat,
        IdentityTransform,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        MultiArmKinematics,
        Resize,
        SimpleStateSampling,
        ToTensor,
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
        use_master_joint=True,
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
        # random_crop_padding = dict(
        #     type=RandomCropPaddingResize,
        #     range_w=(-30, 30),
        #     range_h=(-30, 50),
        #     range_scale=None,
        # )
        # extrinsic_noise = dict(
        #     type=ExtrinsicNoise,
        #     noise_range=(0.04, 0.04, 0.04, 0.015, 0.015, 0.015),
        # )
        transforms = [
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            # random_crop_padding,
            resize,
            to_tensor,
            calib_to_ext,
            # extrinsic_noise,
            ego_to_cam,
            projection_mat,
            scale_shift,
            joint_state_noise,
            convert_dtype,
            kinematics,
            item_selection,
        ]
        from torchvision.transforms import Compose

        from robo_orchard_lab.dataset.robotwin.transforms import ArrowDataParse
        from robo_orchard_lab.utils.build import build
        from robo_orchard_lab.utils.misc import as_sequence

        data_parser = dict(
            type=ArrowDataParse,
            cam_names=["left_hand_camera", "right_hand_camera", "camera_ext"],
            load_image=True,
            load_depth=True,
            load_extrinsic=True,
            depth_scale=1000,
        )
        transforms.insert(0, data_parser)
        transforms = [i for i in transforms if i is not None]
        transforms = Compose([build(x) for x in as_sequence(transforms)])
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
        ]
    return transforms


@train_dataset_register()
def build_datasets(config, dataset_names, mode, lazy_init=True):
    assert mode == "training", "only support training mode"
    from robo_orchard_lab.dataset.robot.dataset import (
        ConcatRODataset,
        ROMultiRowDataset,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import (
        EpisodeSamplerConfig,
    )

    datasets = []
    for dataset_name, data_config in dataset_config.items():
        if (
            "isaac_pick_place" not in dataset_names
            and dataset_name not in dataset_names
        ):
            continue

        # build transforms
        transforms = build_transforms(config, mode, urdf=data_config["urdf"])

        # build dataset sampler
        joint_sampler = EpisodeSamplerConfig(
            target_columns=["joints", "actions"]
        )

        # build dataset
        dataset_list = []
        for data_path in data_config["data_paths"]:
            print(f"Loading arrow dataset from {data_path}...")
            arrow_dataset = ROMultiRowDataset(
                dataset_path=data_path, row_sampler=joint_sampler
            )
            arrow_dataset.set_transform(transforms)
            dataset_list.append(arrow_dataset)
        dataset = ConcatRODataset(dataset_list)
        datasets.append(dataset)

        # viz_arrow_dataset(dataset)

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
            calibration=False,
            depth_restore=False,
            do_calib_to_ext=False,
        )
        processor = SEMProcessor(
            SEMProcessorCfg(
                load_image=True,
                load_depth=config["with_depth_loss"],
                valid_action_step=None,
                cam_names=data_config["cam_names"],
                transforms=transforms,
            )
        )
        processors[dataset_name] = processor
    return processors


def viz_arrow_dataset(dataset):
    import os

    import cv2
    import numpy as np
    from tqdm import tqdm

    from robo_orchard_lab.dataset.robot.db_orm import Episode
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    # Export video to viz
    output_path = "./"
    episode_index = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    file = os.path.join(output_path, "viz.mp4")

    arrow_dataset = dataset.datasets[0]
    episode_info = arrow_dataset.get_meta(Episode, episode_index)
    begin = episode_info.dataset_begin_index
    end = begin + episode_info.frame_num

    for idx in tqdm(range(begin, end)):
        data = arrow_dataset[idx]
        vis_imgs = RoboTwinLmdbDataset.get_vis_imgs(
            data["imgs"],
            data.get("projection_mat"),
            data.get("hist_robot_state", [None])[-1],
        )
        vis_depths = RoboTwinLmdbDataset.depth_visualize(data["depths"])
        vis_depths = np.reshape(
            vis_depths.transpose(1, 0, 2, 3), vis_imgs.shape
        )
        vis_imgs = np.concatenate([vis_imgs, vis_depths], axis=0)
        if idx == begin:
            video_writer = cv2.VideoWriter(
                file, fourcc, 25, vis_imgs.shape[:2][::-1]
            )  # noqa: E501
        video_writer.write(vis_imgs)
    video_writer.release()
