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
    train_dataset_register,
    validation_dataset_register,
)

dataset_config = dict(
    grasp_anything_ro=dict(
        data_paths=[
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_10_29_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_10_30_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_10_31_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_03_xuewu_lin",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_03_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_04_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_05_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_06_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_07_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_10_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_11_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_12_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_13_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_14_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_17_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_18_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_19_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_20_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_21_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_24_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_25_xuewu_lin",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_25_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_26_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_27_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_11_28_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_12_01_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_12_02_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_12_03_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_12_04_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2025_12_05_xuewu_lin",
            "./data/arrow/horizon_beijing/place_objects_to_basket_2026_01_09_zhixu_zhao",
            "./data/arrow/horizon_beijing/place_objects_to_basket_demo_2025_11_25_zhixu_zhao",
        ],
        urdf="./urdf/piper_description_dualarm_new.urdf",
        cam_names=["left", "right", "middle"],
    ),
)


def build_transforms(
    config,
    mode,
    cam_names,
    urdf,
    depth_restore=False,
):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ArrowDataParse,
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

    arrow_data_parser = dict(
        type=ArrowDataParse,
        cam_names=cam_names,
        depth_scale=1000,
        hist_steps=config["hist_steps"],
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
            arrow_data_parser,
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            random_crop_padding,
            resize,
            to_tensor,
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
            arrow_data_parser,
            depth_restoration,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        calib_to_ext = dict(
            type=CalibrationToExtrinsic,
            cam_ee_joint_indices=dict(left=5, right=12),
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
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


@train_dataset_register()
@validation_dataset_register()
def build_datasets(config, dataset_names, mode, **kwargs):
    from torchvision.transforms import Compose

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        EpisodeChunkSamplerConfig,
    )
    from robo_orchard_lab.dataset.robot.dataset import (
        ConcatRODataset,
        ROMultiRowDataset,
    )
    from robo_orchard_lab.utils.build import build
    from robo_orchard_lab.utils.misc import as_sequence

    row_sampler = EpisodeChunkSamplerConfig(
        target_columns=["joints", "actions"],
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )

    datasets = {}
    for data_name, data_config in dataset_config.items():
        if data_name not in dataset_names:
            continue

        transforms = build_transforms(
            config,
            mode,
            cam_names=data_config["cam_names"],
            urdf=data_config["urdf"],
            depth_restore=config.get("depth_restore", False),
        )
        composed_transforms = Compose(
            [build(x) for x in as_sequence(transforms)]
        )

        ro_datasets = []
        for data_path in data_config["data_paths"]:
            ro_dataset = ROMultiRowDataset(
                dataset_path=data_path,
                row_sampler=row_sampler,
            )
            ro_dataset.set_transform(composed_transforms)
            ro_datasets.append(ro_dataset)

        assert len(ro_datasets) > 0, f"No datasets found for {data_name}"

        dataset = ConcatRODataset(ro_datasets)
        dataset.flag = data_config.get(
            "flag",
            int(uuid.uuid5(uuid.NAMESPACE_DNS, "agilex").hex[:4], 16),
        )
        datasets[data_name] = dataset

    return datasets
