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

kinematics_config = dict(
    cfg1=dict(
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
        ee_to_gripper=[
            [
                [
                    -0.9999682379372994,
                    -0.0003115185599020969,
                    0.007964048766817892,
                    0.0018707328290231695,
                ],
                [
                    0.0002697238991081376,
                    -0.9999861904061724,
                    -0.005248451768744473,
                    -0.0018856685642450247,
                ],
                [
                    0.007965573776675915,
                    -0.0052461369728042375,
                    0.9999545128060932,
                    0.28286744101103084,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
    cfg2=dict(
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
        ee_to_gripper=[
            [
                [
                    -0.9999999999997691,
                    -5.018992113630731e-07,
                    -4.5744708197323086e-07,
                    -0.0006724132390504289,
                ],
                [
                    5.018989525192326e-07,
                    -0.9999999999997136,
                    5.659501833159083e-07,
                    -0.0004777270276046519,
                ],
                [
                    -4.574473660118027e-07,
                    5.659499537211986e-07,
                    0.9999999999997349,
                    0.30901697384099314,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
    cfg3=dict(
        urdf="./urdf/rh20t/ur5/urdf/ur5.urdf",
        arm_joint_id=[list(range(6))],
        arm_link_keys=[
            [
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "ee_link",
            ]
        ],
        ee_to_gripper=[
            [
                [
                    -0.002713943065656337,
                    0.012277433350014431,
                    0.9999209464469541,
                    0.25119763364695097,
                ],
                [
                    -0.0024159064921715568,
                    0.9999216303125459,
                    -0.012283998897830235,
                    -0.0025412466253584053,
                ],
                [
                    -0.9999933989326416,
                    -0.0024490535798068207,
                    -0.0026840692438490693,
                    -0.0006492160913159628,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
    cfg4=dict(
        urdf="./urdf/rh20t/ur5/urdf/ur5.urdf",
        arm_joint_id=[list(range(6))],
        arm_link_keys=[
            [
                "shoulder_link",
                "upper_arm_link",
                "forearm_link",
                "wrist_1_link",
                "wrist_2_link",
                "ee_link",
            ]
        ],
        # finger_keys=[
        #     "ee_link",
        # ],
        ee_to_gripper=[
            [
                [
                    -0.004435816872433704,
                    0.010919962535273251,
                    0.9999299288337928,
                    0.2611471763838254,
                ],
                [
                    -0.005368504697076875,
                    0.999925358971501,
                    -0.010943639032521862,
                    -0.0019415587115155853,
                ],
                [
                    -0.9999749784905956,
                    -0.0054164563476855375,
                    -0.0043770494398595785,
                    -0.001164169820553302,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
    cfg5=dict(
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
                "panda_link7",
            ]
        ],
        # finger_keys=[
        #     "panda_leftfinger",
        #     "panda_rightfinger"
        # ],
        ee_to_gripper=[
            [
                [
                    -0.7076175266820617,
                    -0.7065911742703078,
                    -0.0019454342511294877,
                    -6.720066328061262e-05,
                ],
                [
                    0.7065926133386217,
                    -0.7076190196453096,
                    -0.00043476207271402615,
                    -7.592247260656084e-05,
                ],
                [
                    -0.0010696186662914014,
                    -0.0016809409343306674,
                    0.9999965605763904,
                    0.2103617170588283,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
    cfg6=dict(
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
                "lbr_iiwa_link_7",
            ]
        ],
        # finger_keys=[
        #     "lbr_iiwa_link_7",
        # ],
        ee_to_gripper=[
            [
                [
                    0.9915242040032483,
                    -0.1228028699055486,
                    0.00927475482159916,
                    0.0032540648729314373,
                ],
                [
                    0.12307464657373286,
                    0.985421821614017,
                    -0.10413776946492038,
                    -0.001039395901507895,
                ],
                [
                    0.00510715769626161,
                    0.10445746318983228,
                    0.9938860838781031,
                    0.3036302427015539,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
    cfg7=dict(
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
                "lbr_iiwa_link_7",
            ]
        ],
        # finger_keys=[
        #     "lbr_iiwa_link_7",
        # ],
        ee_to_gripper=[
            [
                [
                    0.995777469513335,
                    -0.07689974945457902,
                    0.00813664770461177,
                    0.005159095772118501,
                ],
                [
                    0.07726590170277187,
                    0.9925531775292902,
                    -0.06683514852839949,
                    0.0022704843385951035,
                ],
                [
                    -0.0007654728486078227,
                    0.06725245715640088,
                    0.9967486626004483,
                    0.3027788150581747,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    ),
)


scale_shift_config = dict(
    cfg1=dict(
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
    cfg2=dict(
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
    cfg3=dict(
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
    cfg4=dict(
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
    cfg5=dict(
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
    cfg6=dict(
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
    cfg7=dict(
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

data_paths = dict(
    cfg1=[
        "./data/rh20t/RH20T_cfg1/shard_000",
        "./data/rh20t/RH20T_cfg1/shard_001",
        "./data/rh20t/RH20T_cfg1/shard_002",
        "./data/rh20t/RH20T_cfg1/shard_004",
    ],
    cfg2=[
        "./data/rh20t/RH20T_cfg2/shard_000",
        "./data/rh20t/RH20T_cfg2/shard_001",
    ],
    cfg3=[
        "./data/rh20t/RH20T_cfg3/shard_000",
    ],
    cfg4=[
        "./data/rh20t/RH20T_cfg4/shard_000",
        "./data/rh20t/RH20T_cfg4/shard_001",
        "./data/rh20t/RH20T_cfg4/shard_002",
    ],
    cfg5=[
        "./data/rh20t/RH20T_cfg5/shard_000",
        "./data/rh20t/RH20T_cfg5/shard_001",
    ],
    cfg6=[
        "./data/rh20t/RH20T_cfg6/shard_000",
        "./data/rh20t/RH20T_cfg6/shard_001",
    ],
    cfg7=[
        "./data/rh20t/RH20T_cfg7/shard_000",
    ],
)


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

    if mode == "training":
        add_data_relative_items = AddItems(
            state_loss_weights=state_loss_weights,
            fk_loss_weight=fk_loss_weight,
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
        )
    else:
        add_data_relative_items = AddItems(
            T_base2ego=t_base2ego,
            T_base2world=t_base2world,
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


def build_datasets(config, dataset_names, mode, lazy_init=True):
    assert mode == "training", "only support training mode"
    from robo_orchard_lab.dataset.horizon_manipulation import (
        RH20TManipulationDataset,
    )
    from robo_orchard_lab.dataset.lmdb.instruction_reader import (
        InstructionReader,
    )

    datasets = []
    for dataset_name, paths in data_paths.items():
        if (
            "rh20t" not in dataset_names
            and f"rh20t-{dataset_name}" not in dataset_names
        ):
            continue
        transforms = build_transforms(
            config,
            mode,
            scale_shift_config[dataset_name],
            kinematics_config[dataset_name],
        )
        instruction_reader = dict(
            type=InstructionReader,
            lmdb_path="./data/instructions/subtasks_agibot_rh20t_agilex_20250714/",
            instruction_path="./data/instructions/task2instruction.json",
        )
        dataset = RH20TManipulationDataset(
            paths=paths,
            lazy_init=lazy_init or mode != "training",
            transforms=transforms,
            dataset_name=f"rh20t-{dataset_name}",
            num_views=3,
            instruction_reader=instruction_reader,
        )
        datasets.append(dataset)
    return datasets
