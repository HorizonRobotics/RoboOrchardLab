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

DATA_TYPE = "robotwin"


dataset_config = dict(
    aloha_v1=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/aloha/aloha.urdf",
            arm_link_keys=[
                [
                    "fl_link1",
                    "fl_link2",
                    "fl_link3",
                    "fl_link4",
                    "fl_link5",
                    "fl_link6_ee",
                ],
                [
                    "fr_link1",
                    "fr_link2",
                    "fr_link3",
                    "fr_link4",
                    "fr_link5",
                    "fr_link6_ee",
                ],
            ],
            arm_joint_id=[
                [10, 11, 12, 13, 14, 15],
                [18, 19, 20, 21, 22, 23],
            ],
            finger_keys=[
                ["fl_link6_gripper_end"],
                ["fr_link6_gripper_end"],
            ],
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
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
        ],
        num_joint=14,
        cam_names=[
            "front_camera",
            "left_camera",
            "right_camera",
            "head_camera",
        ],
    ),
    aloha_v2=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/aloha/aloha.urdf",
            arm_link_keys=[
                [
                    "fl_link1",
                    "fl_link2",
                    "fl_link3",
                    "fl_link4",
                    "fl_link5",
                    "fl_link6_ee",
                ],
                [
                    "fr_link1",
                    "fr_link2",
                    "fr_link3",
                    "fr_link4",
                    "fr_link5",
                    "fr_link6_ee",
                ],
            ],
            arm_joint_id=[
                [10, 11, 12, 13, 14, 15],
                [18, 19, 20, 21, 22, 23],
            ],
            finger_keys=[
                ["fl_link6_gripper_end"],
                ["fr_link6_gripper_end"],
            ],
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.5, 0.5],
            [1.12735104, -0.11648428],
            [1.45046443, 1.35436516],
            [1.5324732, 1.45750941],
            [1.80842297, -0.01855904],
            [1.46318083, 0.16631192],
            [2.79637467, 0.24332368],
            [0.5, 0.5],
        ],
        num_joint=14,
        cam_names=[
            "front_camera",
            "left_camera",
            "right_camera",
            "head_camera",
        ],
    ),
    ur5_wsg=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/ur5_wsg/robotwin2_dual_arm_ur5_wsg.urdf",
            arm_link_keys=[
                [
                    "left_shoulder_link",
                    "left_upper_arm_link",
                    "left_forearm_link",
                    "left_wrist_1_link",
                    "left_wrist_2_link",
                    "left_wrist_3_link",
                ],
                [
                    "right_shoulder_link",
                    "right_upper_arm_link",
                    "right_forearm_link",
                    "right_wrist_1_link",
                    "right_wrist_2_link",
                    "right_wrist_3_link",
                ],
            ],
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            finger_keys=[
                ["left_wrist_3_link_gripper_end"],
                ["right_wrist_3_link_gripper_end"],
            ],
        ),
        T_base2world=[
            [1, 0, 0, 0],
            [0, 1, 0, -0.65],
            [0, 0, 1, 0.65],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [2.400281548500061, -0.1310516595840454],
            [1.445511817932129, -1.445511817932129],
            [2.16847026348114, -0.23492777347564697],
            [1.7424615025520325, -0.007538259029388428],
            [2.8101450204849243, 0.15472495555877686],
            [2.9653799533843994, 0.02583003044128418],
            [0.5, 0.5],
            [2.400281548500061, -0.1310516595840454],
            [1.445511817932129, -1.445511817932129],
            [2.16847026348114, -0.23492777347564697],
            [1.7424615025520325, -0.007538259029388428],
            [2.8101450204849243, 0.15472495555877686],
            [2.9653799533843994, 0.02583003044128418],
            [0.5, 0.5],
        ],
        num_joint=14,
        cam_names=["left_camera", "right_camera", "head_camera"],
    ),
    arx_x5a=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/arx_x5a/robotwin2_dual_arm_arx_x5a.urdf",
            arm_link_keys=[
                [
                    "left_link1",
                    "left_link2",
                    "left_link3",
                    "left_link4",
                    "left_link5",
                    "left_link6_ee",
                ],
                [
                    "right_link1",
                    "right_link2",
                    "right_link3",
                    "right_link4",
                    "right_link5",
                    "right_link6_ee",
                ],
            ],
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            finger_keys=[
                ["left_link6_gripper_end"],
                ["right_link6_gripper_end"],
            ],
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.35],
            [0, 0, 1, 0.784],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [1.593699038028717, -0.07424229383468628],
            [1.5048249727115035, 1.4888149732723832],
            [1.5240914672613144, 1.392913356423378],
            [2.3121931552886963, -0.11049866676330566],
            [1.8181148767471313, -0.006529808044433594],
            [2.981711745262146, -0.015958189964294434],
            [0.5, 0.5],
            [1.593699038028717, -0.07424229383468628],
            [1.5048249727115035, 1.4888149732723832],
            [1.5240914672613144, 1.392913356423378],
            [2.3121931552886963, -0.11049866676330566],
            [1.8181148767471313, -0.006529808044433594],
            [2.981711745262146, -0.015958189964294434],
            [0.5, 0.5],
        ],
        num_joint=14,
        cam_names=["left_camera", "right_camera", "head_camera"],
    ),
    franka_panda=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/franka_panda/franka_panda.urdf",
            arm_link_keys=[
                [
                    "panda_left_link1",
                    "panda_left_link2",
                    "panda_left_link3",
                    "panda_left_link4",
                    "panda_left_link5",
                    "panda_left_link6",
                    "panda_left_link7_ee",
                ],
                [
                    "panda_right_link1",
                    "panda_right_link2",
                    "panda_right_link3",
                    "panda_right_link4",
                    "panda_right_link5",
                    "panda_right_link6",
                    "panda_right_link7_ee",
                ],
            ],
            arm_joint_id=[list(range(7)), list(range(9, 16))],
            finger_keys=[
                ["panda_left_link7_gripper_end"],
                ["panda_right_link7_gripper_end"],
            ],
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0.75],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [2.6311211585998535, 0.1413583755493164],
            [1.761036455631256, 0.0017635226249694824],
            [2.374313175678253, -0.4328884482383728],
            [1.2648795694112778, -1.4153200536966324],
            [2.89715313911438, -0.0001468658447265625],
            [1.1023293435573578, 1.897346407175064],
            [2.8941593170166016, 0.002657175064086914],
            [0.5, 0.5],
            [2.6311211585998535, 0.1413583755493164],
            [1.761036455631256, 0.0017635226249694824],
            [2.374313175678253, -0.4328884482383728],
            [1.2648795694112778, -1.4153200536966324],
            [2.89715313911438, -0.0001468658447265625],
            [1.1023293435573578, 1.897346407175064],
            [2.8941593170166016, 0.002657175064086914],
            [0.5, 0.5],
        ],
        num_joint=16,
        cam_names=["left_camera", "right_camera", "head_camera"],
    ),
    piper=dict(
        kinematics_config=dict(
            urdf="./urdf/robotwin/piper/robotwin2_dual_arm_piper.urdf",
            arm_link_keys=[
                [
                    "left_link1",
                    "left_link2",
                    "left_link3",
                    "left_link4",
                    "left_link5",
                    "left_link6_ee",
                ],
                [
                    "right_link1",
                    "right_link2",
                    "right_link3",
                    "right_link4",
                    "right_link5",
                    "right_link6_ee",
                ],
            ],
            arm_joint_id=[list(range(6)), list(range(8, 14))],
            finger_keys=[
                ["left_link6_gripper_end"],
                ["right_link6_gripper_end"],
            ],
        ),
        T_base2world=[
            [0, -1, 0, 0],
            [1, 0, 0, -0.45],
            [0, 0, 1, 0.75],
            [0, 0, 0, 1],
        ],
        scale_shift=[
            [1.2148041427135468, -0.5527651607990265],
            [1.5329843759536743, 1.5329843759536743],
            [1.351477067451924, -1.3455229592509568],
            [1.8320000171661377, 0.0],
            [1.2200000286102295, 0.0],
            [3.1352850198745728, 0.00451505184173584],
            [0.5, 0.5],
            [1.257849007844925, 0.5348821580410004],
            [1.5490366220474243, 1.5490366220474243],
            [1.355304169934243, -1.341695856768638],
            [1.8296949863433838, -0.0023050308227539062],
            [1.2200000286102295, 0.0],
            [3.130632758140564, 0.00936734676361084],
            [0.5, 0.5],
        ],
        num_joint=14,
        cam_names=["left_camera", "right_camera", "head_camera"],
    ),
)


def get_dataset_config():
    """Return the dataset config mapping.

    Provided so the URDF alignment tooling (and other consumers that need a
    single, uniform entry point across dataset families) can invoke a getter
    the same way it does for InternA1. The training / evaluation code paths
    continue to reference the module-level ``dataset_config`` dict directly.
    """

    return dataset_config


def build_transforms(
    config,
    mode,
    kinematics_config,
    t_base2world,
    scale_shift,
    num_joint,
    reference_img_path=None,
):
    import numpy as np

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        MultiArmKinematics,
    )
    from robo_orchard_lab.dataset.robotwin.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        IdentityTransform,
        ImageChannelFlip,
        ItemSelection,
        JointStateNoise,
        LoadReferenceImages,
        MoveEgoToCam,
        Resize,
        SimpleResize,
        SimpleStateSampling,
        ToTensor,
        UnsqueezeBatch,
    )
    from robo_orchard_lab.transforms import ValueSampling

    value_sampling = (
        dict(
            type=ValueSampling,
            norm_mode=config["value_norm_mode"],
            task_max_step=None,
        )
        if config.get("value_model_training", False)
        else None
    )

    with_reference_imgs = reference_img_path is not None and config.get(
        "with_reference_imgs", False
    )
    if with_reference_imgs:
        load_reference_img = dict(
            type=LoadReferenceImages, path=reference_img_path
        )
        reference_img_dst_wh = config.get("reference_img_dst_wh", (224, 224))
        resize_reference_img = dict(
            type=SimpleResize,
            keys="reference_imgs",
            dst_wh=reference_img_dst_wh,
        )
    else:
        load_reference_img = resize_reference_img = dict(
            type=IdentityTransform
        )

    num_joint_per_arm = num_joint // 2 - 1
    joint_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    ee_state_loss_weights = [1, 2, 2, 2, 0.2, 0.2, 0.2, 0.2]
    loss_weights = np.array(
        [
            [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
            + [joint_state_loss_weights] * num_joint_per_arm
            + [ee_state_loss_weights]
        ]
    ).tolist()
    joint_mask = ([True] * num_joint_per_arm + [False]) * 2

    if mode == "training":
        add_data_relative_items = dict(
            type=AddItems,
            T_base2world=t_base2world,
            state_loss_weights=loss_weights,
            fk_loss_weight=loss_weights,
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
    )
    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    img_channel_flip = dict(type=ImageChannelFlip, output_channel=[2, 1, 0])
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

    kinematics = dict(type=MultiArmKinematics, **kinematics_config)

    scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)

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
                "joint_mask",
                "value",
                *(["reference_imgs"] if with_reference_imgs else []),
            ],
        )
        joint_state_noise = dict(
            type=JointStateNoise,
            noise_range=([[-0.02, 0.02]] * num_joint_per_arm + [[0.0, 0.0]])
            * 2,
        )
        transforms = [
            add_data_relative_items,
            load_reference_img,
            resize_reference_img,
            value_sampling,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
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
                "joint_mask",
                "value",
                *(["reference_imgs"] if with_reference_imgs else []),
            ],
        )
        transforms = [
            add_data_relative_items,
            load_reference_img,
            resize_reference_img,
            value_sampling,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
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
                "joint_mask",
                *(["reference_imgs"] if with_reference_imgs else []),
            ],
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            load_reference_img,
            resize_reference_img,
            state_sampling,
            resize,
            img_channel_flip,
            to_tensor,
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
    mode="training",
    lazy_init=True,
    reference_img_path=None,
):
    from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
        RoboTwinLmdbDataset,
    )

    transforms = build_transforms(
        config,
        mode,
        dataset_config[setting_type]["kinematics_config"],
        dataset_config[setting_type]["T_base2world"],
        dataset_config[setting_type]["scale_shift"],
        dataset_config[setting_type]["num_joint"],
        reference_img_path=reference_img_path,
    )
    return RoboTwinLmdbDataset(
        paths=data_paths,
        task_names=config.get("task_names"),
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        cam_names=dataset_config[setting_type]["cam_names"],
        reset_step=1000,
    )


def _build_processor(config, setting_type):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(
        config,
        "deploy",
        dataset_config[setting_type]["kinematics_config"],
        dataset_config[setting_type]["T_base2world"],
        dataset_config[setting_type]["scale_shift"],
        dataset_config[setting_type]["num_joint"],
    )
    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            transforms=transforms,
            cam_names=dataset_config[setting_type]["cam_names"],
        )
    )


@processor_register(DATA_TYPE)
def build_processors(
    config,
    dataset_name,
    setting_type,
):
    return _build_processor(config, setting_type=setting_type)
