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

from collections.abc import Sequence

import numpy as np
from dataset_factory import processor_register, train_dataset_register


def build_block_stratified_episode_row_indices(
    episode_indices: Sequence[int],
    *,
    block_size: int = 12,
    ratio: float = 1.0,
    seed: int = 42,
) -> list[int]:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if ratio <= 0 or ratio > 1:
        raise ValueError("ratio must be in the range (0, 1].")
    if ratio == 1.0:
        return list(range(len(episode_indices)))

    episode_to_rows = {}
    ordered_episodes = []
    for row_idx, episode_idx in enumerate(episode_indices):
        if episode_idx not in episode_to_rows:
            episode_to_rows[episode_idx] = []
            ordered_episodes.append(episode_idx)
        episode_to_rows[episode_idx].append(row_idx)

    rng = np.random.default_rng(seed)
    selected_rows = []
    for start in range(0, len(ordered_episodes), block_size):
        block = ordered_episodes[start : start + block_size]
        keep_count = min(
            len(block),
            max(1, int(round(len(block) * ratio))),
        )
        selected_positions = rng.permutation(len(block))[:keep_count]
        for pos in selected_positions:
            selected_rows.extend(episode_to_rows[block[pos]])

    return sorted(selected_rows)


class EpisodeSubsetDataset:
    def __init__(
        self,
        dataset,
        row_indices: Sequence[int],
        *,
        dataset_name: str | None = None,
    ):
        self.dataset = dataset
        self.row_indices = list(row_indices)
        self.dataset_name = dataset_name or getattr(
            dataset, "dataset_name", "unnamed"
        )

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.dataset[self.row_indices[index]]
        if isinstance(index, list):
            return self.dataset[[self.row_indices[i] for i in index]]
        if isinstance(index, slice):
            return self[list(range(*index.indices(len(self))))]
        raise TypeError(f"Unsupported index type: {type(index)!r}")

    @property
    def features(self):
        return self.dataset.features

    @property
    def transform(self):
        return self.dataset.transform

    def set_transform(self, transform):
        self.dataset.set_transform(transform)

    @property
    def meta_index2meta(self):
        return self.dataset.meta_index2meta

    @meta_index2meta.setter
    def meta_index2meta(self, value):
        self.dataset.meta_index2meta = value


DATA_TYPE = "isaac"

ROBOT_PROFILES = dict(
    piperx=dict(
        urdf="./urdf/piper_x_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
        joint_mask=([True] * 6 + [False]) * 2,
        gripper_indices=[6, 13],
        joint_state_loss_weights=[1, 0, 0, 0, 0, 0, 0, 0],
        ee_state_loss_weights=[1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1],
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
        joint_state_noise_range=[
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
        kinematics_config=dict(
            urdf="./urdf/piper_x_description_dualarm.urdf",
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
        calib_to_ext_kwargs=dict(cam_ee_joint_indices=dict(left=5, right=12)),
        arrow_cam_names=[
            "left_hand_camera",
            "right_hand_camera",
            "static_camera",
        ],
    ),
    franka=dict(
        urdf="/horizon-bucket/robot_lab/assets/ROBOTS/FRANKA/franka_panda.urdf",
        cam_names=["ext1_camera", "ext2_camera", "wrist_camera"],
        joint_mask=[True] * 7 + [False],
        gripper_indices=[7],
        joint_state_loss_weights=[1, 0, 0, 0, 0, 0, 0, 0],
        ee_state_loss_weights=[1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1],
        scale_shift=[
            [0.3342398234502883, 0.05480703378777375],
            [0.34801173277799424, 0.14486869546571998],
            [0.25697947733358306, -0.009624386876346059],
            [0.3041917461078475, -2.1549022377761684],
            [0.48351280064879965, 0.03227452103317364],
            [0.3164671897776108, 1.9411939409982757],
            [0.42373574881002624, 0.8050510254576515],
            [0.02240243915587483, 0.06110932271404834],
        ],
        joint_state_noise_range=[
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.02, 0.02],
            [-0.0, 0.0],
        ],
        kinematics_config=dict(
            urdf="/horizon-bucket/robot_lab/assets/ROBOTS/FRANKA/franka_panda.urdf",
            arm_joint_id=[list(range(7))],
            arm_link_keys=[
                [
                    "panda_link1",
                    "panda_link2",
                    "panda_link3",
                    "panda_link4",
                    "panda_link5",
                    "panda_link6",
                    "panda_link7",
                ]
            ],
            finger_keys=[["panda_leftfinger"]],
        ),
        calib_to_ext_kwargs=dict(cam_ee_joint_indices=dict(wrist_camera=5)),
        arrow_cam_names=["ext1_camera", "ext2_camera", "wrist_camera"],
    ),
)

dataset_config = dict(
    isaac_pick_place=dict(
        robot_type="piperx",
    )
)


def get_robot_profile(setting_type):
    return ROBOT_PROFILES[dataset_config[setting_type]["robot_type"]]


def build_transforms(
    config,
    mode,
    urdf,
    robot_profile,
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
        UnsqueezeBatch,
    )

    if depth_restore:
        depth_restoration = dict(type=DepthRestoration)
    else:
        depth_restoration = dict(type=IdentityTransform)

    t_base2world = np.eye(4).tolist()  # noqa: N806
    joint_mask = robot_profile["joint_mask"]

    joint_state_loss_weights = robot_profile["joint_state_loss_weights"]
    ee_state_loss_weights = robot_profile["ee_state_loss_weights"]
    arm_joint_ids = robot_profile["kinematics_config"]["arm_joint_id"]
    loss_weight_tokens = []
    for joint_ids in arm_joint_ids:
        loss_weight_tokens.extend([joint_state_loss_weights] * len(joint_ids))
        loss_weight_tokens.append(ee_state_loss_weights)
    loss_weights = np.array([loss_weight_tokens])
    state_loss_weights = loss_weights * 0.2
    fk_loss_weight = loss_weights * 1.8
    state_loss_weights = state_loss_weights.tolist()
    fk_loss_weight = fk_loss_weight.tolist()

    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        use_master_gripper=True,
        use_master_joint=True,
        gripper_indices=robot_profile["gripper_indices"],
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

    kinematics_config = dict(robot_profile["kinematics_config"])
    kinematics_config["urdf"] = urdf
    kinematics = dict(type=MultiArmKinematics, **kinematics_config)

    if do_calib_to_ext:
        calib_to_ext = dict(
            type=CalibrationToExtrinsic,
            calibration=calibration,
            **robot_profile.get("calib_to_ext_kwargs", {}),
            **kinematics_config,
        )
    else:
        calib_to_ext = dict(type=IdentityTransform)

    scale_shift = dict(
        type=AddScaleShift,
        scale_shift=robot_profile["scale_shift"],
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
                # "hist_joint_state", # for pi
                # "pred_joint_state", # for pi
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
            noise_range=robot_profile["joint_state_noise_range"],
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
        if config.get("openpi", False):
            transforms = [
                state_sampling,
                resize,
                to_tensor,
                joint_state_noise,
                item_selection,
            ]
            print("Using openpi transforms")

        from torchvision.transforms import Compose

        from robo_orchard_lab.dataset.robotwin.transforms import ArrowDataParse
        from robo_orchard_lab.utils.build import build
        from robo_orchard_lab.utils.misc import as_sequence

        data_parser = dict(
            type=ArrowDataParse,
            cam_names=robot_profile.get(
                "arrow_cam_names", robot_profile["cam_names"]
            ),
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
def build_datasets(
    config,
    dataset_name,
    data_paths,
    setting_type,
    mode,
    lazy_init=True,
):
    assert mode == "training", "only support training mode"
    from robo_orchard_lab.dataset.robot.dataset import (
        ConcatRODataset,
        ROMultiRowDataset,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import (
        EpisodeSamplerConfig,
    )

    robot_profile = get_robot_profile(setting_type)
    transforms = build_transforms(
        config,
        mode,
        urdf=robot_profile["urdf"],
        robot_profile=robot_profile,
    )

    joint_sampler = EpisodeSamplerConfig(target_columns=["joints", "actions"])

    dataset_list = []
    for data_path_idx, data_path in enumerate(data_paths):
        print(f"Loading arrow dataset from {data_path}...")
        arrow_dataset = ROMultiRowDataset(
            dataset_path=data_path,
            row_sampler=joint_sampler,
            meta_index2meta=True,
        )
        arrow_dataset.set_transform(transforms)

        data_ratio = config.get("data_ratio", 1.0)
        if data_ratio < 1.0:
            row_indices = build_block_stratified_episode_row_indices(
                arrow_dataset.index_dataset["episode_index"],
                block_size=config.get("data_subset_block_size", 12),
                ratio=data_ratio,
                seed=config.get("data_subset_seed", 42) + data_path_idx,
            )
            print(
                "Using "
                f"{len(row_indices)}/{len(arrow_dataset)} rows "
                f"from {data_path} with data_ratio={data_ratio}."
            )
            arrow_dataset = EpisodeSubsetDataset(
                arrow_dataset,
                row_indices,
                dataset_name=f"{dataset_name}_{data_path_idx}",
            )
        dataset_list.append(arrow_dataset)
    datasets = ConcatRODataset(dataset_list)

    return datasets


@processor_register(DATA_TYPE)
def build_processors(config, dataset_name, setting_type, **kwargs):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    robot_profile = get_robot_profile(setting_type)
    transforms = build_transforms(
        config,
        mode="deploy",
        urdf=robot_profile["urdf"],
        robot_profile=robot_profile,
        calibration=False,
        depth_restore=False,
        do_calib_to_ext=False,
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth_loss"],
            valid_action_step=None,
            cam_names=robot_profile["cam_names"],
            transforms=transforms,
        )
    )
