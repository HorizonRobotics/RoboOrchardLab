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


from dataset_factory import train_dataset_register

DATA_TYPE = "rh20t"

kinematics_config = dict(
    flexiv=dict(
        urdf="./urdf/rh20t/flexiv/robot_pvt3.urdf",
        arm_joint_id=[list(range(7))],
        arm_link_keys=[
            [
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
            ]
        ],
        finger_keys=[["link7_gripper_end"]],
    ),
    flexiv_v2=dict(
        urdf="./urdf/rh20t/flexiv_v2/robot_pvt3.urdf",
        arm_joint_id=[list(range(7))],
        arm_link_keys=[
            [
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
            ]
        ],
        finger_keys=[["link7_gripper_end"]],
    ),
    ur5=dict(
        urdf="./urdf/rh20t/ur5/ur5.urdf",
        arm_joint_id=[list(range(6))],
        arm_link_keys=[
            [
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
            ]
        ],
        finger_keys=[["wrist_3_link_gripper_end"]],
    ),
    ur5_v2=dict(
        urdf="./urdf/rh20t/ur5_v2/ur5.urdf",
        arm_joint_id=[list(range(6))],
        arm_link_keys=[
            [
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "wrist_3_link",
            ]
        ],
        finger_keys=[["wrist_3_link_gripper_end"]],
    ),
    franka=dict(
        urdf="./urdf/rh20t/franka/franka_womaterial.urdf",
        arm_joint_id=[list(range(7))],
        arm_link_keys=[
            [
                "panda_link1",
                "panda_link2",
                "panda_link3",
                "panda_link4",
                "panda_link5",
                "panda_link6",
                "panda_link7_ee",
            ]
        ],
        finger_keys=[["panda_link7_gripper_end"]],
    ),
    kuka=dict(
        urdf="./urdf/rh20t/kuka/model.urdf",
        arm_joint_id=[list(range(7))],
        arm_link_keys=[
            [
                "lbr_iiwa_link_1",
                "lbr_iiwa_link_2",
                "lbr_iiwa_link_3",
                "lbr_iiwa_link_4",
                "lbr_iiwa_link_5",
                "lbr_iiwa_link_6",
                "lbr_iiwa_link_7_ee",
            ]
        ],
        finger_keys=[["lbr_iiwa_link_7_gripper_end"]],
    ),
    kuka_v2=dict(
        urdf="./urdf/rh20t/kuka_v2/model.urdf",
        arm_joint_id=[list(range(7))],
        arm_link_keys=[
            [
                "lbr_iiwa_link_1",
                "lbr_iiwa_link_2",
                "lbr_iiwa_link_3",
                "lbr_iiwa_link_4",
                "lbr_iiwa_link_5",
                "lbr_iiwa_link_6",
                "lbr_iiwa_link_7_ee",
            ]
        ],
        finger_keys=[["lbr_iiwa_link_7_gripper_end"]],
    ),
)


scale_shift_config = dict(
    flexiv=dict(
        scale_shift=[
            [1.8854153972405654, 0.12018490846340468],
            [1.3257280804894187, -0.3932210423729636],
            [2.0904658351625716, -0.16401653630392898],
            [1.4666698770597577, 1.2457420034334064],
            [2.1101008103444023, -0.6303306267811701],
            [2.0260400587114793, 0.7678624206575855],
            [2.944754583495004, 0.09540249620165153],
            [0.0475, 0.0475],
        ],
    ),
    flexiv_v2=dict(
        scale_shift=[
            [1.8854153972405654, 0.12018490846340468],
            [1.3257280804894187, -0.3932210423729636],
            [2.0904658351625716, -0.16401653630392898],
            [1.4666698770597577, 1.2457420034334064],
            [2.1101008103444023, -0.6303306267811701],
            [2.0260400587114793, 0.7678624206575855],
            [2.944754583495004, 0.09540249620165153],
            [0.0475, 0.0475],
        ],
    ),
    ur5=dict(
        scale_shift=[
            [1.0607173832563253, -0.28546532071553743],
            [0.9490752716859181, -1.3848951955636342],
            [1.274408912229848, 1.3335016047046688],
            [1.6517929855514975, -1.4769359651733847],
            [1.4805675426764149, -1.5500084003167494],
            [2.105875566777061, -1.0206552417839276],
            [0.055, 0.055],
        ]
    ),
    ur5_v2=dict(
        scale_shift=[
            [1.0607173832563253, -0.28546532071553743],
            [0.9490752716859181, -1.3848951955636342],
            [1.274408912229848, 1.3335016047046688],
            [1.6517929855514975, -1.4769359651733847],
            [1.4805675426764149, -1.5500084003167494],
            [2.105875566777061, -1.0206552417839276],
            [0.055, 0.055],
        ]
    ),
    franka=dict(
        scale_shift=[
            [0.7346614707052761, -0.047866176420455486],
            [0.992782645312074, 0.2626264762012926],
            [0.20986790088825097, -0.149406397064401],
            [1.1428362824366545, -1.8022382758214022],
            [1.8568252159349297, 0.08777582764318759],
            [1.1382067501544952, 1.9992835223674774],
            [1.7505874918442634, 1.1540723131638435],
            [0.04041, 0.040400000000000005],
        ]
    ),
    kuka=dict(
        scale_shift=[
            [0.8286045863166479, -0.013757557099485895],
            [0.7971857621181788, 0.6024209452866816],
            [0.12276526139577214, 0.07080681546277819],
            [0.9902254641310464, -1.1025460369652713],
            [1.5151476297001374, -0.08009155181016925],
            [0.987248343640381, 1.1117668372690612],
            [2.111909314583902, 0.8981051382148777],
            [0.0425, 0.0425],
        ]
    ),
    kuka_v2=dict(
        scale_shift=[
            [0.8286045863166479, -0.013757557099485895],
            [0.7971857621181788, 0.6024209452866816],
            [0.12276526139577214, 0.07080681546277819],
            [0.9902254641310464, -1.1025460369652713],
            [1.5151476297001374, -0.08009155181016925],
            [0.987248343640381, 1.1117668372690612],
            [2.111909314583902, 0.8981051382148777],
            [0.0425, 0.0425],
        ]
    ),
)


def get_kinematics_config():
    return kinematics_config


def build_transforms(config, mode, scale_shift, kinematics_config):
    import numpy as np
    import torch

    from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
        AddItems,
        AddScaleShift,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        MultiArmKinematics,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UpSampleJointState,
    )

    t_base2ego = np.eye(4)
    t_base2world = np.eye(4)

    joint_state_loss_weights = [1, 0, 0, 0, 0, 0, 0, 0]
    ee_state_loss_weights = [1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1]
    num_joint = len(scale_shift["scale_shift"]) - 1
    loss_weights = np.array(
        [[joint_state_loss_weights] * num_joint + [ee_state_loss_weights]]
    )
    rh20t_loss_weight = 0.5
    state_loss_weights = loss_weights * 0.2 * rh20t_loss_weight
    fk_loss_weight = loss_weights * 1.8 * rh20t_loss_weight
    joint_mask = [True] * num_joint + [False]

    if mode == "training":
        add_data_relative_items = AddItems(
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )
    else:
        add_data_relative_items = AddItems(
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
            joint_mask=joint_mask,
        )

    state_sampling = SimpleStateSampling(
        hist_steps=config["hist_steps"] // 3 + 1,
        pred_steps=config["pred_steps"] // 3 + 1,
        use_master_gripper=True,
        use_master_joint=False,
        gripper_indices=[6, 13],
    )

    joint_upsample = UpSampleJointState(
        pred_steps=config["pred_steps"],
        hist_steps=config["hist_steps"],
    )

    resize = Resize(
        dst_wh=config.get("dst_wh", (308, 252)),
    )
    to_tensor = ToTensor()
    projection_mat = GetProjectionMat(target_coordinate="ego")
    convert_dtype = ConvertDataType(
        convert_map=dict(
            imgs=torch.float32,
            depths=torch.float32,
            image_wh=torch.float32,
            projection_mat=torch.float32,
            embodiedment_mat=torch.float32,
        )
    )

    kinematics = MultiArmKinematics(**kinematics_config)

    scale_shift = AddScaleShift(**scale_shift)
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
                "subtask",
                "joint_mask",
            ]
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            joint_upsample,
            projection_mat,
            scale_shift,
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
                "subtask",
                "joint_mask",
            ]
        )
        transforms = [
            add_data_relative_items,
            state_sampling,
            resize,
            to_tensor,
            joint_upsample,
            projection_mat,
            scale_shift,
            convert_dtype,
            kinematics,
            item_selection,
        ]
    elif mode == "deploy":
        raise NotImplementedError
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
    from robo_orchard_lab.dataset.horizon_manipulation import (
        RH20TManipulationDataset,
    )
    from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
        InstructionReader,
    )

    transforms = build_transforms(
        config,
        mode,
        scale_shift_config[setting_type],
        kinematics_config[setting_type],
    )
    instruction_reader = InstructionReader(
        paths="./data/instructions_v2/rh20t"
    )
    return RH20TManipulationDataset(
        paths=data_paths,
        lazy_init=lazy_init or mode != "training",
        transforms=transforms,
        dataset_name=dataset_name,
        num_views=3,
        instruction_reader=instruction_reader,
    )
