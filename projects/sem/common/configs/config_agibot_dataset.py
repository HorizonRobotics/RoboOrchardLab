import glob
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


data_paths = dict(
    agibot_alpha=[
        "./data/agibot/AgiBotWorld-Alpha-250414_34fd7cd4_lmdb",
    ],
    agibot_beta=[
        "./data/agibot/AgiBotWorld-Beta-250412-2496da_lmdb",
    ],
)

kinematics_config = dict(
    agibot=dict(
        urdf="./urdf/G1_120s_dual.urdf",
        left_arm_joint_id=list(range(7)),  # AgiBot: 7 joints per arm
        right_arm_joint_id=list(
            range(8, 15)
        ),  # AgiBot: joints 8-14 for right arm
        left_arm_link_keys=[
            "Joint1_l",
            "Joint2_l",
            "Joint3_l",
            "Joint4_l",
            "Joint5_l",
            "Joint6_l",
            "Joint7_l",
        ],
        right_arm_link_keys=[
            "Joint1_r",
            "Joint2_r",
            "Joint3_r",
            "Joint4_r",
            "Joint5_r",
            "Joint6_r",
            "Joint7_r",
        ],
        left_finger_keys=["Joint7_l"],  # AgiBot uses Joint7 as end effector
        right_finger_keys=["Joint7_r"],
        use_ee_state=True,
    )
)

agibot_scale_shift = [
    # Left arm joints (7 joints)
    [0.807397993, -1.075758200],  # Joint1_l
    [0.519545991, 0.628004746],  # Joint2_l
    [0.827265625, 0.755998523],  # Joint3_l
    [0.431041460, -0.748906466],  # Joint4_l
    [0.357815969, 0.392597735],  # Joint5_l
    [0.389363049, 1.321284795],  # Joint6_l
    [0.477344665, 0.298801437],  # Joint7_l
    [33.615918755, 54.628037467],  # left_gripper
    # Right arm joints (7 joints)
    [1.056550615, 0.897950653],  # Joint1_r
    [0.610640659, -0.593406163],  # Joint2_r
    [0.892552863, -0.728802693],  # Joint3_r
    [0.413018960, 0.736117755],  # Joint4_r
    [0.567352174, -0.411271768],  # Joint5_r
    [0.313294437, -1.254856470],  # Joint6_r
    [0.490895931, -0.434084028],  # Joint7_r
    [34.681226171, 61.714828429],  # right_gripper
    # Head joints (2 joints)
    [0.049485552, -0.027264017],  # joint_head_pitch
    [0.077976995, 0.451097382],  # joint_head_yaw
    # Body joints (2 joints)
    [0.082625759, 0.430035826],  # joint_body_pitch
    [0.076923081, 0.274950588],  # joint_lift_body
]


def find_agibot_shards(dataset_root):
    """Find all valid AgiBot shard directories."""
    all_shard_paths = glob.glob(os.path.join(dataset_root, "*", "shard_*"))

    valid_paths = []

    for path in all_shard_paths:
        valid = True
        for subdir in ["index", "meta", "depth", "image"]:
            if not os.path.isfile(os.path.join(path, subdir, "data.mdb")):
                valid = False
                logger.warning(
                    f"Skipping invalid shard: {path}, missing '{subdir}/data.mdb'"
                )
                break
        if valid:
            valid_paths.append(path)
    logger.info(f"Found {len(valid_paths)} valid AgiBot dataset shards.")
    valid_paths.sort()
    return valid_paths


def build_transforms(config):
    from robo_orchard_lab.dataset.agibot.transforms import (
        AddItems,
        AddScaleShift,
        AgiBotDualArmKinematics,
        ConvertDataType,
        GetProjectionMat,
        ItemSelection,
        JointSelection,
        Resize,
        SimpleStateSampling,
        ToTensor,
        UpSampleJointState,
    )

    state_sampling = SimpleStateSampling(
        hist_steps=config["hist_steps"] // 3 + 1,
        pred_steps=config["pred_steps"] // 3 + 1,
        gripper_indices=[7, 15],  # AgiBot gripper indices
    )

    joint_upsample = UpSampleJointState(
        pred_steps=config["pred_steps"],
        hist_steps=config["hist_steps"],
    )

    dst_wh = config.get("dst_wh", (308, 252))
    resize = Resize(dst_wh=dst_wh)

    to_tensor = ToTensor()
    projection_mat = GetProjectionMat(target_coordinate="base")
    convert_dtype = ConvertDataType(
        convert_map=dict(
            imgs=torch.float32,
            depths=torch.float32,
            image_wh=torch.float32,
            projection_mat=torch.float32,
            embodiedment_mat=torch.float32,
            pred_mask=torch.bool,
        )
    )

    # AgiBot dual-arm kinematics configuration (shared for alpha and beta)
    kinematics = AgiBotDualArmKinematics(**kinematics_config["agibot"])

    scale_shift = AddScaleShift(
        scale_shift=(
            agibot_scale_shift
            if not config.get("relative")
            else [
                # Relative mode scale_shift for 20 joints
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (0.015, 0),  # left arm + gripper
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (0.015, 0),  # right arm + gripper
                (1, 0),
                (1, 0),  # head joints
                (1, 0),
                (1, 0),  # body joints
            ]
        ),
    )

    item_selection = ItemSelection(
        keys=[
            "imgs",
            "depths",
            "image_wh",
            "projection_mat",
            "embodiedment_mat",
            "hist_robot_state",
            # "hist_ee_state",
            "pred_robot_state",
            "joint_relative_pos",
            "joint_scale_shift",
            "kinematics",
            "text",
            "uuid",
            "pred_mask",
            "subtask",
            "state_loss_weights",
            "fk_loss_weight",
        ]
    )

    # Joint selection to filter to dual-arm joints only (16 out of 20)
    joint_selection = JointSelection(selection_mode="no_head")

    base_joint_weights = [
        1,
    ] + [0.0] * 7  # Regular arm joints: pos > rot
    lift_weights = [
        1,
    ] + [0.0] * 7  # Lift joints: same as base
    gripper_weights = [
        1,
        1,
        1,
        1,
        0.1,
        0.1,
        0.1,
        0.1,
    ]  # Gripper joints: same as base
    l1, l2 = 0.2, 1.8
    loss_weight = np.array(
        [
            [base_joint_weights] * 7  # left arm (7)
            + [gripper_weights] * 1  # left gripper (1)
            + [base_joint_weights] * 7  # right arm (7)
            + [gripper_weights] * 1  # right gripper (1)
            + [lift_weights] * 2  # lift joints (2)
        ]
    )  # 1, num_joint, 8

    add_data_relative_items = AddItems(
        state_loss_weights=loss_weight * l1,
        fk_loss_weight=loss_weight * l2,
        T_base2ego=np.eye(4),
        T_base2world=np.eye(4),
    )
    return [
        add_data_relative_items,
        state_sampling,
        resize,
        to_tensor,
        joint_upsample,
        projection_mat,
        scale_shift,
        convert_dtype,
        kinematics,
        joint_selection,  # Filter robot_state and joint_scale_shift to 16 joints
        item_selection,
    ]


def build_datasets(
    config,
    dataset_names,
    interval=None,
    lazy_init=True,
    lmdb_kwargs=None,
    mode="training",
):
    """Build AgiBot datasets for training."""
    assert mode == "training", "only support training mode"
    valid = False
    for name in ["agibot", "agibot_alpha", "agibot_beta"]:
        if name in dataset_names:
            valid = True
            break
    if not valid:
        return []
    from robo_orchard_lab.dataset.agibot.agibot_lmdb_dataset import (
        AgiBotLmdbDataset,
    )
    from robo_orchard_lab.dataset.lmdb.instruction_reader import (
        InstructionReader,
    )

    train_datasets = []
    transforms = build_transforms(config)
    for name, dataset_roots in data_paths.items():
        if "agibot" not in dataset_names and name not in dataset_names:
            continue

        shard_paths = []
        for dataset_root in dataset_roots:
            shard_paths.extend(find_agibot_shards(dataset_root))

        instruction_reader = dict(
            type=InstructionReader,
            lmdb_path="./data/instructions/subtasks_agibot_rh20t_agilex_20250714/",
            instruction_path="./data/instructions/task2instruction.json",
        )
        agibot_dataset = AgiBotLmdbDataset(
            paths=shard_paths,
            lazy_init=lazy_init,
            transforms=transforms,
            cam_names=None,
            dataset_name=name,
            task_info_reader=instruction_reader,
        )
        train_datasets.append(agibot_dataset)
    return train_datasets
