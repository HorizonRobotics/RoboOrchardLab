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

# ruff: noqa: E501

import copy
import os
from glob import glob

DATA_BASE = os.environ.get("HOLOBRAIN_DATA_BASE", "./data")


def _glob_sorted(*patterns: str) -> list[str]:
    data_paths = []
    for pattern in patterns:
        data_paths.extend(glob(pattern))
    return sorted(set(data_paths))


TRAINING_DATASETS = [
    # # ================ robotwin ===================
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin1_0",
        setting_type="aloha_v1",
        data_paths=[f"{DATA_BASE}/robotwin1.0"],
    ),
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0",
        setting_type="aloha_v2",
        data_paths=[
            f"{DATA_BASE}/robotwin2.0/aloha_agilex_demo_clean",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part1",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part2",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part3",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part4",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part5",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part6",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part7",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part8",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part9",
            f"{DATA_BASE}/robotwin2.0/agilex_demo_randomized_500_part10",
        ],
    ),
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0_ur5_wsg",
        setting_type="ur5_wsg",
        data_paths=[f"{DATA_BASE}/robotwin2.0/ur5_wsg_demo_clean"],
    ),
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0_arx_x5a",
        setting_type="arx_x5a",
        data_paths=[f"{DATA_BASE}/robotwin2.0/arx-x5_demo_clean"],
    ),
    dict(
        dataset_type="robotwin",
        dataset_name="robotwin2_0_franka_panda",
        setting_type="franka_panda",
        data_paths=[f"{DATA_BASE}/robotwin2.0/franka-panda_demo_clean"],
    ),
    # dict(  # without text instruction
    #     dataset_type="robotwin",
    #     dataset_name="robotwin2_0_piper",
    #     setting_type="piper",
    #     data_paths=[
    #         f"{DATA_BASE}/robotwin2.0/aloha_piper_27tasks_clean_200",
    #         f"{DATA_BASE}/robotwin2.0/aloha_piper_27tasks_noise_300",
    #     ],
    # ),
    # ================ agilex ===================
    dict(
        dataset_type="agilex",
        dataset_name="challenge",
        setting_type="challenge",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/challenge/倒水*/*",
            f"{DATA_BASE}/challenge/叠毛巾*/叠毛巾-白黑格纹/*",
            f"{DATA_BASE}/challenge/叠毛巾*/叠毛巾517mm*",
            f"{DATA_BASE}/challenge/叠盘子*/*",
            f"{DATA_BASE}/challenge/叠短裤*/*",
            f"{DATA_BASE}/challenge/盖笔帽*/*",
        ),
    ),
    dict(
        dataset_type="agilex",
        dataset_name="challenge_finetune",
        setting_type="challenge_finetune",
        data_paths=[f"{DATA_BASE}/challenge/finetune"],
    ),
    dict(
        dataset_type="agilex",
        dataset_name="challenge_self_collect",
        setting_type="challenge_self_collect",
        data_paths=[
            f"{DATA_BASE}/challenge/agilex_data_0527_plates_stack",
            f"{DATA_BASE}/challenge/agilex_data_0525",
        ],
    ),
    dict(
        dataset_type="agilex",
        dataset_name="horizon_beijing",
        setting_type="horizon_beijing",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-empty_cup_place",
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-collect_bottles",
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-collect_bottles-20250707-v2",
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-place_to_slot",
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-place_to_slot-20250709",
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-two_fold_towel-20250710",
            f"{DATA_BASE}/horizon_beijing/xuewu.lin-two_fold_towel-20250712",
            f"{DATA_BASE}/horizon_beijing/zhixu.zhao-*",
            f"{DATA_BASE}/horizon_beijing/*-place_objects_to_basket-*",
            f"{DATA_BASE}/horizon_beijing/*-fold_paper_box-*",
        ),
    ),
    dict(
        dataset_type="agilex",
        dataset_name="horizon_beijing_piper_x",
        setting_type="horizon_beijing_piper_x",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/horizon_beijing/*-piper_x-*-*"
        ),
    ),
    dict(
        dataset_type="agilex",
        dataset_name="horizon_shanghai",
        setting_type="horizon_shanghai",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/horizon_shanghai/*-empty_cup_place-*",
            f"{DATA_BASE}/horizon_shanghai/*-place_shoe-*",
            f"{DATA_BASE}/horizon_shanghai/*-place_to_slot-*",
            f"{DATA_BASE}/horizon_shanghai/*-put_bottles_dustbin-*",
            f"{DATA_BASE}/horizon_shanghai/*-stack_block_two-*",
            f"{DATA_BASE}/horizon_shanghai/*-stack_bowls_three-*",
            f"{DATA_BASE}/horizon_shanghai/*-two_fold_towel-*",
            f"{DATA_BASE}/horizon_shanghai/*-fold_clothes-*",
            f"{DATA_BASE}/horizon_shanghai/*-flatten_clothes-*",
            f"{DATA_BASE}/horizon_shanghai/*-place_object_to_location-*",
        ),
    ),
    dict(
        dataset_type="agilex",
        dataset_name="agilex",
        setting_type="agilex",
        data_paths=lambda: _glob_sorted(f"{DATA_BASE}/agilex/lmdb/*"),
    ),
    # ================ agibot ===================
    dict(
        dataset_type="agibot",
        dataset_name="agibot",
        data_paths=lambda: _glob_sorted(f"{DATA_BASE}/agibot/valid_lmdbs/*"),
        instruction_path=f"{DATA_BASE}/instructions_v2/agibot",
    ),
    dict(
        dataset_type="agibot_geniesim",
        dataset_name="agibot_geniesim3_challenge",
        data_paths=[
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/hold_pot",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/open_door",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/place_block_into_box",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/clean_the_desktop*",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/pour_workpiece",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/scoop_popcorn*",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/sorting_packages_part*",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/stock_and_straighten_shelf*",
            f"{DATA_BASE}/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim-with_depth/take_wrong_item_shelf",
        ],
    ),
    # ================ droid ===================
    dict(
        dataset_type="droid",
        dataset_name="droid",
        data_paths=[
            f"{DATA_BASE}/droid/RAIL/success",
            f"{DATA_BASE}/droid/REAL/success",
            f"{DATA_BASE}/droid/AUTOLab/success",
            f"{DATA_BASE}/droid/GuptaLab/success",
            f"{DATA_BASE}/droid/IRIS/success",
            f"{DATA_BASE}/droid/TRI/success",
            f"{DATA_BASE}/droid/CLVR/success",
            f"{DATA_BASE}/droid/ILIAD/success",
            f"{DATA_BASE}/droid/IPRL/success",
            f"{DATA_BASE}/droid/PennPAL/success",
            f"{DATA_BASE}/droid/RPL/success",
        ],
    ),
    # ================ egodex ===================
    dict(
        dataset_type="egodex",
        dataset_name="egodex",
        data_paths=[
            f"{DATA_BASE}/egodex/lmdb/part1",
            f"{DATA_BASE}/egodex/lmdb/part2",
            f"{DATA_BASE}/egodex/lmdb/part3",
            f"{DATA_BASE}/egodex/lmdb/part4",
            f"{DATA_BASE}/egodex/lmdb/part5",
            f"{DATA_BASE}/egodex/lmdb/extra",
            f"{DATA_BASE}/egodex/lmdb/test",
        ],
    ),
    # ================ interna1 ===================
    dict(
        dataset_type="interna1",
        dataset_name="interna1_arx_lift2",
        setting_type="interna1_arx_lift2",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/InternData_A1_lmdb/ARX_Lift2/lmdb_dataset_ARX_Lift2*"
        ),
    ),
    dict(
        dataset_type="interna1",
        dataset_name="interna1_agile_split_aloha",
        setting_type="interna1_agile_split_aloha",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/InternData_A1_lmdb/AgileX_Split_Aloha/"
            "lmdb_dataset_AgileX_Split_Aloha*"
        ),
    ),
    # ================ libero ===================
    dict(
        dataset_type="libero",
        dataset_name="libero_goal",
        data_paths=f"{DATA_BASE}/libero/lmdb_goal_abs",
    ),
    dict(
        dataset_type="libero",
        dataset_name="libero_object",
        data_paths=f"{DATA_BASE}/libero/lmdb_object_abs",
    ),
    dict(
        dataset_type="libero",
        dataset_name="libero_spatial",
        data_paths=f"{DATA_BASE}/libero/lmdb_spatial_abs",
    ),
    dict(
        dataset_type="libero",
        dataset_name="libero_10",
        data_paths=f"{DATA_BASE}/libero/lmdb_10_abs",
    ),
    # ================ table30v2 ===================
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_ur5",
        setting_type="ur5",
        data_paths=[
            f"{DATA_BASE}/table30v2/lmdb/ur5/arrange_fruits",
            f"{DATA_BASE}/table30v2/lmdb/ur5/item_classification",
            f"{DATA_BASE}/table30v2/lmdb/ur5/shred_paper",
        ],
    ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_arx5",
        setting_type="arx5",
        data_paths=[
            f"{DATA_BASE}/table30v2/lmdb/arx5/arrange_flowers",
            f"{DATA_BASE}/table30v2/lmdb/arx5/hang_the_cup",
            f"{DATA_BASE}/table30v2/lmdb/arx5/pick_out_the_green_blocks",
            # f"{DATA_BASE}/table30v2/lmdb/arx5/press_the_button",
            f"{DATA_BASE}/table30v2/lmdb/arx5/turn_on_the_light_switch",
            f"{DATA_BASE}/table30v2/lmdb/arx5/water_the_flowers",
            f"{DATA_BASE}/table30v2/lmdb/arx5/wipe_the_table",
        ],
    ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_aloha",
        setting_type="aloha",
        data_paths=[
            f"{DATA_BASE}/table30v2/lmdb/aloha/put_the_books_back",
            f"{DATA_BASE}/table30v2/lmdb/aloha/stamp_positioning",
            f"{DATA_BASE}/table30v2/lmdb/aloha/wipe_the_blackboard",
            f"{DATA_BASE}/table30v2/lmdb/aloha/scoop_with_a_small_spoon",
            f"{DATA_BASE}/table30v2/lmdb/aloha/lint_roller_remove_dirt",
            f"{DATA_BASE}/table30v2/lmdb/aloha/pack_the_items",
            f"{DATA_BASE}/table30v2/lmdb/aloha/pack_the_toothbrush_holder",
            f"{DATA_BASE}/table30v2/lmdb/aloha/paint_jam",
            f"{DATA_BASE}/table30v2/lmdb/aloha/put_the_pencil_case_into_the_schoolbag",
            f"{DATA_BASE}/table30v2/lmdb/aloha/wrap_with_a_soft_cloth",
        ],
    ),
    dict(
        dataset_type="table30v2",
        dataset_name="table30v2_dos_w1",
        setting_type="dos_w1",
        data_paths=[
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/fold_the_clothes",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/stack_bowls",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/hold_the_tray_with_both_hands",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/place_objects_into_desk_drawer",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/tidy_up_the_makeup_table",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/put_in_pen_container",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/put_the_shoes_back",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/sweep_the_trash",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/tie_a_knot",
            f"{DATA_BASE}/table30v2/lmdb/dos_w1/untie_the_shoelaces",
        ],
    ),
    # ================== behavior =======================
    dict(
        dataset_type="behavior",
        dataset_name="behavior_manipulation",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/behavior1k_lmdb_data/task_*_manipulation_lmdb"
        ),
    ),
    dict(
        dataset_type="behavior",
        dataset_name="behavior_navigation",
        data_paths=lambda: _glob_sorted(
            f"{DATA_BASE}/behavior1k_lmdb_data/task_*_navigation_lmdb"
        ),
    ),
    # ================ rh20t ===================
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_flexiv",
        setting_type="flexiv",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg1/shard_000",
            f"{DATA_BASE}/rh20t/RH20T_cfg1/shard_001",
            f"{DATA_BASE}/rh20t/RH20T_cfg1/shard_002",
            f"{DATA_BASE}/rh20t/RH20T_cfg1/shard_004",
        ],
    ),
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_flexiv_v2",
        setting_type="flexiv_v2",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg2/shard_000",
            f"{DATA_BASE}/rh20t/RH20T_cfg2/shard_001",
        ],
    ),
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_ur5",
        setting_type="ur5",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg3/shard_000",
        ],
    ),
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_ur5_v2",
        setting_type="ur5_v2",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg4/shard_000",
            f"{DATA_BASE}/rh20t/RH20T_cfg4/shard_001",
            f"{DATA_BASE}/rh20t/RH20T_cfg4/shard_002",
        ],
    ),
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_franka",
        setting_type="franka",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg5/shard_000",
            f"{DATA_BASE}/rh20t/RH20T_cfg5/shard_001",
        ],
    ),
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_kuka",
        setting_type="kuka",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg6/shard_000",
            f"{DATA_BASE}/rh20t/RH20T_cfg6/shard_001",
        ],
    ),
    dict(
        dataset_type="rh20t",
        dataset_name="rh20t_kuka_v2",
        setting_type="kuka_v2",
        data_paths=[
            f"{DATA_BASE}/rh20t/RH20T_cfg7/shard_000",
        ],
    ),
    # ================= isaac ===================
    dict(
        dataset_type="isaac",
        dataset_name="isaac_pick_place",
        setting_type="isaac_pick_place",
        data_paths=[
            "/horizon-bucket/robot_lab/users/mengao.zhao/dataset/pick_place_arrow/stack_block_two_seed0-499",
            "/horizon-bucket/robot_lab/users/mengao.zhao/dataset/pick_place_arrow/place_mouse_pad_seed0-499",
            "/horizon-bucket/robot_lab/users/mengao.zhao/dataset/pick_place_arrow/place_lemon_plate_seed0-499",
        ],
    ),
    # ================== agibot digit ===============
    dict(
        dataset_type="agibot_digit",
        dataset_name="agibot_digit",
        data_paths=[
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/pack_in_the_supermarket",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/clear_table_in_the_restaurant",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/stamp_the_seal",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/restock_supermarket_items",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/clear_the_countertop_waste",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/open_drawer_and_store_items",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/make_a_sandwich",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/heat_the_food_in_the_microwave",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/pack_moving_objects_from_conveyor",
            f"{DATA_BASE}/arrow/agibot_digit_challenge/2026_0125_lite/pickup_items_from_the_freezer",
        ],
        adc_anno_results_dir=f"{DATA_BASE}/arrow/agibot_digit_challenge/adc_anno_results",
    ),
    # =================== agilex ro ===================
    dict(
        dataset_type="agilex_ro",
        dataset_name="agilex_ro",
        setting_type="grasp_anything_ro",
        data_paths=[
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_10_29_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_10_30_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_10_31_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_03_xuewu_lin",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_03_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_04_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_05_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_06_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_07_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_10_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_11_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_12_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_13_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_14_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_17_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_18_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_19_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_20_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_21_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_24_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_25_xuewu_lin",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_25_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_26_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_27_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_11_28_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_12_01_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_12_02_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_12_03_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_12_04_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2025_12_05_xuewu_lin",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_2026_01_09_zhixu_zhao",
            f"{DATA_BASE}/arrow/horizon_beijing/place_objects_to_basket_demo_2025_11_25_zhixu_zhao",
        ],
    ),
    # ================ table30 ro ===================
    dict(
        dataset_type="table30_ro",
        dataset_name="table30_v1_aloha_ro",
        data_paths=[
            f"{DATA_BASE}/arrow/20260305_01/table30/make_vegetarian_sandwich",
            f"{DATA_BASE}/arrow/20260305_01/table30/clean_dining_table",
            f"{DATA_BASE}/arrow/20260305_01/table30/pour_fries_into_plate",
            f"{DATA_BASE}/arrow/20260305_01/table30/put_opener_in_drawer",
            f"{DATA_BASE}/arrow/20260305_01/table30/stack_bowls",
            f"{DATA_BASE}/arrow/20260305_01/table30/turn_on_faucet",
            f"{DATA_BASE}/arrow/20260305_01/table30/plug_in_network_cable",
        ],
    ),
]

# TODO
VALIDATION_DATASETS = None

training_datasets = copy.deepcopy(TRAINING_DATASETS)
training_datasets.sort(key=lambda x: x["dataset_name"])

filter_list = [
    "robotwin1_0",
    "robotwin2_0",
    "robotwin2_0_ur5_wsg",
    "robotwin2_0_arx_x5a",
    "robotwin2_0_franka_panda",
    # "robotwin2_0_piper",
    "challenge",
    "challenge_finetune",
    "challenge_self_collect",
    "horizon_beijing",
    "horizon_beijing_piper_x",
    "horizon_shanghai",
    "agilex",
    "agibot",
    "agibot_geniesim3_challenge",
    "droid",
    "egodex",
    "interna1_arx_lift2",
    "interna1_agile_split_aloha",
    "libero_goal",
    "libero_object",
    "libero_spatial",
    "libero_10",
    "table30v2_ur5",
    "table30v2_arx5",
    "table30v2_aloha",
    "table30v2_dos_w1",
    "rh20t_flexiv",
    "rh20t_flexiv_v2",
    "rh20t_ur5",
    "rh20t_ur5_v2",
    "rh20t_franka",
    "rh20t_kuka",
    "rh20t_kuka_v2",
    "behavior_manipulation",
    "behavior_navigation",
    # "isaac_pick_place",
    # "agibot_digit",
    # "agilex_ro",
    # "table30_ro",
]

dataset_sample_weights = dict(
    robotwin1_0=0.8,
    robotwin2_0=3,
    robotwin2_0_ur5_wsg=1,
    robotwin2_0_arx_x5a=1,
    robotwin2_0_franka_panda=1,
    robotwin2_0_piper=1,
    challenge=2,
    challenge_finetune=0.1,
    challenge_self_collect=0.15,
    horizon_beijing=8,
    horizon_beijing_piper_x=0.001,
    horizon_shanghai=8,
    agilex=12,
    agibot=10,
    agibot_geniesim3_challenge=2,
    droid=5,
    egodex=10,
    interna1_arx_lift2=10,
    interna1_agile_split_aloha=10,
    libero_goal=0.1,
    libero_object=0.1,
    libero_spatial=0.1,
    libero_10=0.2,
    table30v2_ur5=2,
    table30v2_arx5=2,
    table30v2_aloha=2,
    table30v2_dos_w1=2,
    rh20t_flexiv=0.1,
    rh20t_flexiv_v2=0.1,
    rh20t_ur5=0.1,
    rh20t_ur5_v2=0.1,
    rh20t_franka=0.1,
    rh20t_kuka=0.1,
    rh20t_kuka_v2=0.1,
    behavior_manipulation=5,
    behavior_navigation=0,
)

use_dataset_sample_weights = True

training_datasets = [
    x for x in training_datasets if x["dataset_name"] in filter_list
]
if use_dataset_sample_weights:
    training_datasets = [
        x
        for x in training_datasets
        if dataset_sample_weights[x["dataset_name"]] > 0
    ]
    for x in training_datasets:
        x["sample_weight"] = dataset_sample_weights[x["dataset_name"]]

validation_datasets = copy.deepcopy(VALIDATION_DATASETS)
