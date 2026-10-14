# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

"""Panda Droid action LMDB configuration for HoloBrain 0.5."""

from typing import TYPE_CHECKING

from projects.holobrain_0_5.configs.robot_specs import get_robot_spec

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.horizon_manipulation.horizon_manipulation_dataset import (  # noqa: E501
        HorizonManipulationLmdbDataset,
    )

__all__ = ["build_transforms", "build_processors", "build_datasets"]

DATA_TYPE = "isaac"

dataset_config = dict(
    panda_droid=dict(
        robot_type="isaac_panda_droid",
        cam_names=["wrist_camera", "ext1_camera", "ext2_camera"],
        joint_state_loss_weights=[1, 0, 0, 0, 0, 0, 0, 0],
        ee_state_loss_weights=[1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1],
        scale_shift=[
            [0.380729927895, -0.129912261691],
            [0.612020915840, -0.177131810533],
            [0.299044168732, 0.090433894900],
            [0.348029880707, -2.212067262245],
            [0.413506814718, 0.084409451697],
            [0.402862741569, 2.085753339642],
            [0.562038144803, 0.154399819215],
            [0.402326689383, 0.282474907326],
        ],
        joint_state_noise_range=[[-0.02, 0.02]] * 7 + [[0.0, 0.0]],
        calib_to_ext_kwargs=dict(
            cam_ref_links=dict(
                ext1_camera=None,
                ext2_camera=None,
                wrist_camera=None,
            )
        ),
        arrow_cam_names=["wrist_camera", "ext1_camera", "ext2_camera"],
        resize_config=dict(dst_wh=(448, 256)),
    ),
)
"""Panda profile shared with the Isaac Arrow configuration, not a reader."""


def build_transforms(config: dict, mode: str, setting_type: str) -> list[dict]:
    """Configure Panda observation/action preprocessing for HoloBrain 0.5.

    Raw samples contain ordered BGR views, metric depth, camera intrinsics,
    base-to-camera matrices under ``T_world2cam``, and an episode joint
    trajectory. History uses observations; future targets use
    ``master_joint_state`` when present. FK emits seven arm tokens and one
    TCP/gripper token, each containing joint value, xyz and wxyz.

    Args:
        config (dict): Model settings including hist_steps, pred_steps and
            with_depth; load_image/load_depth control training input loading.
        mode (str): training adds joint noise and loss weights; validation
            omits those; deploy adds a size-one batch dimension.
        setting_type (str): Registered data profile, currently panda_droid.

    Returns:
        list[dict]: Ordered transform construction configs. Image/depth
            decoding and camera ordering belong to the reader or processor.

    Raises:
        ValueError: If mode is not training, validation or deploy.
        KeyError: If the data profile is unregistered.
    """
    if mode not in ("training", "validation", "deploy"):
        raise ValueError(f"Unsupported transform mode: {mode}")

    import numpy as np

    from projects.holobrain_0_5.configs.transforms import (
        AddItems,
        ConvertDataType,
        MultiArmKinematics,
        Resize,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        GetProjectionMat,
        ItemSelection,
        JointStateNoise,
        MoveEgoToCam,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import AddScaleShift

    robot_profile = dataset_config[setting_type]
    spec = get_robot_spec(robot_profile["robot_type"])
    # Packed camera poses already use the robot base as their world frame.
    add_items = dict(
        type=AddItems,
        T_base2world=np.eye(4).tolist(),
        joint_mask=spec["joint_mask"],
        arm_length=spec["arm_length"],
    )
    if mode == "training":
        loss_tokens = []
        for joint_ids in spec["kinematics"]["arm_joint_id"]:
            loss_tokens.extend(
                [robot_profile["joint_state_loss_weights"]] * len(joint_ids)
            )
            loss_tokens.append(robot_profile["ee_state_loss_weights"])
        loss_weights = np.array([loss_tokens])
        add_items.update(
            state_loss_weights=(loss_weights * 0.2).tolist(),
            fk_loss_weight=(loss_weights * 1.8).tolist(),
        )

    load_image = True if mode == "deploy" else config.get("load_image", True)
    load_depth = (
        config["with_depth"]
        if mode == "deploy"
        else config.get("load_depth", True)
    )
    transforms = [
        add_items,
        dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            use_master_gripper=True,
            use_master_joint=True,
            gripper_indices=spec["gripper_indices"],
        ),
        dict(
            type=Resize,
            **robot_profile["resize_config"],
            strict_image=load_image,
            strict_depth=load_depth,
        ),
        dict(type=ToTensor),
        dict(type=MoveEgoToCam),
        dict(type=GetProjectionMat, target_coordinate="ego"),
        dict(type=AddScaleShift, scale_shift=robot_profile["scale_shift"]),
    ]
    if mode == "training":
        transforms.append(
            dict(
                type=JointStateNoise,
                noise_range=robot_profile["joint_state_noise_range"],
                add_to_pred=True,
            )
        )
    transforms.extend(
        [
            dict(
                type=ConvertDataType,
                convert_map=dict(
                    arm_length="float32",
                    imgs="float32",
                    depths="float32",
                    image_wh="float32",
                    projection_mat="float32",
                    embodiedment_mat="float32",
                ),
            ),
            dict(
                type=MultiArmKinematics,
                urdf=spec["urdf"],
                **spec["kinematics"],
            ),
        ]
    )
    keys = [
        "imgs",
        "depths",
        "image_wh",
        "projection_mat",
        "embodiedment_mat",
        "hist_robot_state",
        "joint_scale_shift",
        "kinematics",
        "arm_length",
        "text",
        "joint_mask",
    ]
    if mode != "deploy":
        keys.extend(["pred_robot_state", "uuid", "subtask", "pred_mask"])
    if mode == "training":
        keys.extend(["fk_loss_weight", "state_loss_weights"])
    transforms.append(dict(type=ItemSelection, keys=keys))
    if mode == "deploy":
        transforms.append(dict(type=UnsqueezeBatch))
    return transforms


def build_processors(config: dict, dataset_name: str, setting_type: str):
    """Build a standalone Panda processor without changing default exports.

    Args:
        config (dict): Model settings consumed by build_transforms.
        dataset_name (str): Factory selection name; naming/export is owned by
            the caller and does not change the processor's embodiment.
        setting_type (str): Data profile selecting the robot and camera order.

    Returns:
        HoloBrainProcessor: A 0.5 processor with size-one batched output.
            The exact registered pruned URDF must be available to construct it.
    """
    from robo_orchard_lab.models.holobrain_0_5 import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            cam_names=dataset_config[setting_type]["cam_names"],
            transforms=build_transforms(config, "deploy", setting_type),
        )
    )


def build_datasets(
    config: dict,
    dataset_name: str,
    data_paths: str | list[str],
    setting_type: str,
    mode: str = "training",
    lazy_init: bool = True,
    interval: int | None = None,
) -> "HorizonManipulationLmdbDataset":
    """Build an action LMDB reader with the selected Panda transforms.

    Args:
        config (dict): Model settings from the dataset factory.
        dataset_name (str): Selection name from dataset_specs.
        data_paths (str | list[str]): Action LMDB root directory or
            directories.
        setting_type (str): Panda data profile selecting body/camera order.
        mode (str, optional): Factory mode. Default is training.
        lazy_init (bool, optional): Defer opening the reader until access.
            Default is True. Transform construction still requires its URDF.
        interval (int, optional): Frame sampling stride. Default is None.

    Returns:
        HorizonManipulationLmdbDataset: Reader with observed history, master
            action targets and metric-depth preprocessing.

    Raises:
        ValueError: If mode is neither training nor validation.
    """
    if mode not in ("training", "validation"):
        raise ValueError(f"Unsupported dataset mode: {mode}")

    from robo_orchard_lab.dataset.horizon_manipulation.horizon_manipulation_dataset import (  # noqa: E501
        HorizonManipulationLmdbDataset,
    )

    return HorizonManipulationLmdbDataset(
        paths=data_paths,
        transforms=build_transforms(config, mode, setting_type),
        interval=interval,
        lazy_init=lazy_init,
        dataset_name=dataset_name,
        task_names=config.get("task_names"),
        load_image=config.get("load_image", True),
        load_depth=config.get("load_depth", True),
        cam_names=dataset_config[setting_type]["cam_names"],
        load_extrinsic=True,
        load_calibration=False,
        depth_scale=1000,
        reset_step=500,
    )
