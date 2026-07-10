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

from dataset_factory import (
    processor_register,
    train_dataset_register,
    validation_dataset_register,
)

DATA_TYPE = "robocasa"
CAMERA_NAMES = [
    ["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"],
    ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"],
]


def build_transforms(config, mode):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        MoveEgoToCam,
        Resize,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.dataset.robocasa.transforms import (
        SimpleStateSampling,
        TransformRobotState,
    )

    loss_weights = np.array([[[1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]]])
    temporal_weights = np.linspace(1.0, 0.1, config["pred_steps"])[
        :, None, None
    ]
    loss_weights = (loss_weights * temporal_weights).tolist()

    add_data_relative_items = dict(
        type=AddItems,
        joint_mask=(False,),
        joint_relative_pos=((0,),),
        joint_scale_shift=((0.5, 0.5),),
        noise_type="local_joint_local_pose",
        depths=np.zeros([3, 2, 2]).tolist(),  # 3 cameras, fake size: [2, 2]
    )
    if mode == "training":
        add_data_relative_items.update(
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
        )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    to_tensor = dict(type=ToTensor)
    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(type=GetProjectionMat, target_coordinate="ego")

    convert_map = dict(
        imgs="float32",
        image_wh="float32",
        depths="float32",
        projection_mat="float32",
        embodiedment_mat="float32",
        hist_robot_state="float32",
        pred_robot_state="float32",
        joint_scale_shift="float32",
        joint_relative_pos="float32",
    )
    convert_dtype = dict(type=ConvertDataType, convert_map=convert_map)

    item_keys = [
        "imgs",
        "image_wh",
        "depths",
        "projection_mat",
        "embodiedment_mat",
        "hist_robot_state",
        "pred_robot_state",
        "joint_scale_shift",
        "joint_relative_pos",
        "text",
        "uuid",
        "joint_mask",
        "noise_type",
        "pred_mask",
    ]
    if mode == "training":
        item_keys.extend(["state_loss_weights", "fk_loss_weight"])
    elif mode == "deploy":
        item_keys.remove("uuid")
        item_keys.remove("pred_mask")
    elif mode != "validation":
        raise NotImplementedError(f"Unsupported RoboCasa mode: {mode}")

    transform_robot_state = dict(type=TransformRobotState)
    item_selection = dict(type=ItemSelection, keys=item_keys)
    transforms = [
        add_data_relative_items,
        state_sampling,
        resize,
        to_tensor,
        ego_to_cam,
        projection_mat,
        transform_robot_state,
        convert_dtype,
        item_selection,
    ]
    if mode == "deploy":
        transforms.append(dict(type=UnsqueezeBatch))
    return transforms


@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    mode="training",
    lazy_init=True,
    mobile=None,
    mimicgen=None,
):
    from robo_orchard_lab.dataset.robocasa.robocasa_lmdb_dataset import (
        RoboCasaLmdbDataset,
    )

    return RoboCasaLmdbDataset(
        paths=data_paths,
        transforms=build_transforms(config, mode),
        lazy_init=lazy_init or mode != "training",
        dataset_name=dataset_name,
        reset_step=500,
        load_depth=False,
        cam_names=CAMERA_NAMES,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        mobile=mobile,
        mimicgen=mimicgen,
    )


@processor_register(DATA_TYPE)
def build_processors(config, dataset_name, **kwargs):
    from robo_orchard_lab.models.holobrain.processor import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=False,
            valid_action_step=None,
            transforms=build_transforms(config, "deploy"),
            cam_names=CAMERA_NAMES[0],
        )
    )
