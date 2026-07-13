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

DATA_TYPE = "agibot_digit"


def parse_arrow_dirs(base_dir: str, limit: int = -1) -> list[str]:
    from pathlib import Path

    from tqdm import tqdm

    base_path = Path(base_dir)
    if not base_path.is_dir():
        return []

    matches = [
        str(p.parent)
        for p in tqdm(
            base_path.rglob("state.json"),
            desc="Searching arrow",
            disable=True,
        )
        if p.is_file()
    ]
    matches = sorted(set(matches))

    if limit >= 0:
        matches = matches[:limit]

    return matches


def load_adc_anno_results(results_dir: str):
    """
    adc_anno_results_data
    load from json
    {
        'ep0003_AgibotDigitChallenge_2810097_12043049.mp4': [475],
        'ep0004_AgibotDigitChallenge_2810097_12043652.mp4': [455],
        'ep0000_AgibotDigitChallenge_2810051_12030289.mp4': [440, 590],
        'ep0001_AgibotDigitChallenge_2810051_12030418.mp4': [560, 705],
        'ep0002_AgibotDigitChallenge_2810051_12030695.mp4': [480, 645],
    }
    convert to
    {
        'AgibotDigitChallenge/2810097/12043049': [475],
        'AgibotDigitChallenge/2810097/12043652': [455],
        'AgibotDigitChallenge/2810051/12030289': [440, 590],
        'AgibotDigitChallenge/2810051/12030418': [560, 705],
        'AgibotDigitChallenge/2810051/12030695': [480, 645],
    }
    """
    import json
    from glob import glob
    from pathlib import Path

    results_path = Path(results_dir)
    if not results_path.is_dir():
        return {}

    result_files = glob(str(results_path / "*.json"))
    results = {}
    for result_file in result_files:
        with open(result_file, "r") as f:
            data = json.load(f)
            results.update(data)

    adc_anno_results = {}
    for video_name, frame_steps in results.items():
        name = video_name.removesuffix(".mp4")
        uuid = "/".join(name.removesuffix(".mp4").split("_")[1:])
        adc_anno_results[uuid] = frame_steps

    return adc_anno_results


scale_shift = [
    [2.607510076, -0.159500003],  # idx21_arm_l_joint1
    [1.869209991, -0.105399966],  # idx22_arm_l_joint2
    [2.792910085, -0.054700017],  # idx23_arm_l_joint3
    [1.446860061, -0.013549984],  # idx24_arm_l_joint4
    [2.535259949, 0.316050053],  # idx25_arm_l_joint5
    [1.660759912, 0.039550006],  # idx26_arm_l_joint6
    [2.740159975, -0.037450075],  # idx27_arm_l_joint7
    [54.007208334, 54.007198334],  # gripper_hand_l NOTE: [use_state]
    [2.675459848, 0.179849982],  # idx61_arm_r_joint1
    [1.896710025, 0.188099980],  # idx62_arm_r_joint2
    [2.866760002, -0.172349930],  # idx63_arm_r_joint3
    [1.483109937, -0.001499951],  # idx64_arm_r_joint4
    [3.095309959, -0.001699924],  # idx65_arm_r_joint5
    [1.739909993, -0.000100017],  # idx66_arm_r_joint6
    [3.064209924, -0.029600024],  # idx67_arm_r_joint7
    [54.093611227, 54.093601227],  # gripper_hand_r NOTE: [use_state]
    [0.049485552, -0.027264017],  # idx11_head_joint1
    [0.098760010, 0.360549986],  # idx12_head_joint2
    [0.220359997, 0.239650011],  # idx01_body_joint1
    [0.062409983, 0.495999992],  # idx02_body_joint2
]


cam_names = ["left", "right", "middle"]
g1_kinematics_config = dict(
    urdf="./urdf/agibot_digital/g1_omnipicker/g1_omnipicker.urdf",
    arm_joint_id=[
        list(range(4, 11)),
        list(range(19, 26)),
        [2, 3],
        [0, 1],
    ],
    arm_link_keys=[
        [
            "arm_l_link1",
            "arm_l_link2",
            "arm_l_link3",
            "arm_l_link4",
            "arm_l_link5",
            "arm_l_link6",
            "arm_l_end_link",
        ],
        [
            "arm_r_link1",
            "arm_r_link2",
            "arm_r_link3",
            "arm_r_link4",
            "arm_r_link5",
            "arm_r_link6",
            "arm_r_end_link",
        ],
        [
            "head_link1",
            "head_link2",
        ],
        [
            "body_link1",
            "body_link2",
        ],
    ],
    finger_keys=[
        ["arm_l_end_link_gripper_end"],
        ["arm_r_end_link_gripper_end"],
        [],
        [],
    ],
    arm_connection_joint_indices=[
        [None, None, None, 0],
        [None, None, None, 0],
        [None, None, None, 0],
        [0, 0, 0, None],
    ],
)


def get_kinematics_config():
    """Return the digit kinematics dict consumed by the align tool.
    """

    return {"agibot_digit": g1_kinematics_config}


def build_transforms(
    config,
    mode,
    cam_names,
    kinematics_config,
    scale_shift,
    adc_anno_results=None,
):
    import numpy as np

    from robo_orchard_lab.dataset.agibot_digit.transforms import (
        ArrowDataParse,
        SimpleStateSampling,
    )
    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        MoveEgoToCam,
        MultiArmKinematics,
        Resize,
        ToTensor,
        UnsqueezeBatch,
    )

    arrow_data_parser = dict(
        type=ArrowDataParse,
        cam_names=cam_names,
        load_image=True,
        load_depth=True,
        load_extrinsic=True,
        depth_scale=1000,
        use_detailed_instruction=True,  # NOTE: detailed instruction
    )

    base_joint_weights = [1] + [1] * 3 + [0.1] * 4
    body_weights = [1] + [0] * 7
    head_weights = [0] + [0] * 7
    gripper_weights = [1] + [1] * 3 + [0.1] * 4
    loss_weights = np.array(
        [
            [base_joint_weights] * 7
            + [gripper_weights] * 1
            + [base_joint_weights] * 7
            + [gripper_weights] * 1
            + [head_weights] * 2
            + [body_weights] * 2,
        ]
    )
    agibot_digit_loss_weight = 1.0
    state_loss_weights = loss_weights * 0.2 * agibot_digit_loss_weight
    fk_loss_weight = loss_weights * 1.8 * agibot_digit_loss_weight
    joint_mask = ([True] * 7 + [False]) * 2 + [True, True] + [False, True]

    add_data_relative_items = dict(
        type=AddItems,
        state_loss_weights=state_loss_weights.tolist(),
        fk_loss_weight=fk_loss_weight.tolist(),
        joint_mask=np.array(joint_mask).tolist(),
    )

    state_sampling = dict(
        type=SimpleStateSampling,
        hist_steps=config["hist_steps"],
        pred_steps=config["pred_steps"],
        pred_steps_sampling_rate=config.get("pred_steps_sampling_rate", 1),
        gripper_indices=[7, 15],
        limitation=2 * 3.14 * 100,
        use_master_joint=False,
        use_master_gripper=False,
        check_adc_frames=True,  # NOTE: check adc frames
        adc_skip_frames_prev=25,
        adc_skip_frames_next=25,
        adc_anno_results_data=adc_anno_results,
    )

    resize = dict(
        type=Resize,
        dst_wh=config.get("dst_wh", (308, 252)),
    )

    to_tensor = dict(type=ToTensor)

    ego_to_cam = dict(type=MoveEgoToCam)
    projection_mat = dict(
        type=GetProjectionMat,
        target_coordinate="ego",
    )

    add_scale_shift = dict(type=AddScaleShift, scale_shift=scale_shift)

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

    kinematics = dict(
        type=MultiArmKinematics,
        **kinematics_config,
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
                "step_index",
                "T_world2cam",
                "intrinsic",
                "joint_relative_pos",
                "pred_mask",
                "joint_mask",
            ],
        )
        transforms = [
            arrow_data_parser,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            add_scale_shift,
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
                "step_index",
                "T_world2cam",
                "intrinsic",
                "joint_relative_pos",
                "joint_mask",
            ],
        )
        transforms = [
            arrow_data_parser,
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            add_scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        add_data_relative_items = dict(
            type=AddItems,
            joint_mask=np.array(joint_mask).tolist(),
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
                "joint_relative_pos",
                "joint_mask",
            ],
        )
        state_sampling = dict(
            type=SimpleStateSampling,
            hist_steps=config["hist_steps"],
            pred_steps=config["pred_steps"],
            pred_steps_sampling_rate=1,
            gripper_indices=[7, 15],
            limitation=2 * 3.14 * 100,
            use_master_joint=False,
            use_master_gripper=False,
            check_adc_frames=False,
        )
        unsqueeze_batch = dict(type=UnsqueezeBatch)
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            ego_to_cam,
            projection_mat,
            add_scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
            unsqueeze_batch,
        ]
    return transforms


@train_dataset_register(DATA_TYPE)
@validation_dataset_register(DATA_TYPE)
def build_datasets(
    config,
    dataset_name,
    data_paths,
    adc_anno_results_dir,
    mode="training",
    **kwargs,
):

    from torchvision.transforms import Compose

    from robo_orchard_lab.dataset.robot.dataset import (
        ConcatRODataset,
        RODataset,
    )
    from robo_orchard_lab.utils.build import build
    from robo_orchard_lab.utils.misc import as_sequence

    adc_anno_results = load_adc_anno_results(adc_anno_results_dir)
    transforms = build_transforms(
        config,
        mode,
        cam_names=cam_names,
        kinematics_config=g1_kinematics_config,
        scale_shift=scale_shift,
        adc_anno_results=adc_anno_results,
    )
    composed_transforms = Compose([build(x) for x in as_sequence(transforms)])

    resolved_paths = []
    for path in data_paths:
        resolved_paths.extend(parse_arrow_dirs(path))

    ro_datasets = []
    for data_path in resolved_paths:
        ro_dataset = RODataset(dataset_path=data_path, meta_index2meta=False)
        ro_dataset.frame_dataset = ro_dataset.frame_dataset.with_format(
            type="numpy",
            columns=[
                "joint_state",
                "joint_action",
                "joint_raw_frame_index",
                "camera_intrinsics/middle",
                "camera_intrinsics/left",
                "camera_intrinsics/right",
            ],
            output_all_columns=True,
        )
        ro_dataset.set_transform(composed_transforms)
        ro_datasets.append(ro_dataset)

    assert len(ro_datasets) > 0, f"No datasets found for {dataset_name}"
    return ConcatRODataset(ro_datasets)


@processor_register(DATA_TYPE)
def build_processors(config, dataset_name, **kwargs):
    from robo_orchard_lab.models.holobrain import (
        HoloBrainProcessor,
        HoloBrainProcessorCfg,
    )

    transforms = build_transforms(
        config,
        mode="deploy",
        cam_names=cam_names,
        kinematics_config=g1_kinematics_config,
        scale_shift=scale_shift,
    )

    return HoloBrainProcessor(
        HoloBrainProcessorCfg(
            load_image=True,
            load_depth=config["with_depth"],
            valid_action_step=None,
            transforms=transforms,
            cam_names=cam_names,
        )
    )
