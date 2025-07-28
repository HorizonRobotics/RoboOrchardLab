#!/usr/bin/env python3
"""脚本用于统计AgiBot数据集中joint的分布并计算scale和shift参数
用于更新AddScaleShift配置

Usage:
    cd python/robo_orchard_lab/robo_orchard_lab/dataset/agibot/
    python compute_joint_statistics.py
"""

import glob
import logging
import os
import pickle

import lmdb
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_agibot_shards(dataset_root):
    """找到所有有效的AgiBot shard目录."""
    all_shard_paths = glob.glob(os.path.join(dataset_root, "*", "shard_*"))

    if not all_shard_paths:
        logger.error(
            f"No 'shard_*' subdirectories found under '{dataset_root}'"
        )
        return []

    valid_paths = []
    for path in all_shard_paths:
        index_db_file = os.path.join(path, "index", "data.mdb")
        meta_db_file = os.path.join(path, "meta", "data.mdb")

        if os.path.isfile(index_db_file) and os.path.isfile(meta_db_file):
            valid_paths.append(path)
        else:
            logger.warning(f"Skipping invalid shard: {path}")

    logger.info(f"Found {len(valid_paths)} valid AgiBot dataset shards.")
    return valid_paths


def collect_joint_data_from_lmdb(
    dataset_root,
    max_shards=None,
    max_episodes_per_shard=None,
    max_timesteps_per_episode=50,
):
    """从LMDB文件中收集joint数据以统计大量样本

    Args:
        dataset_root: AgiBot数据集根目录
        max_shards: 最大使用的shard数量 (None表示使用所有)
        max_episodes_per_shard: 每个shard最大episode数量 (None表示使用所有)
        max_timesteps_per_episode: 每个episode最大时间步数

    Returns:
        joint_data: [N, 20] numpy array, N是总样本数，20是joint数量
    """
    # 找到所有shard
    data_paths = find_agibot_shards(dataset_root)
    if not data_paths:
        raise ValueError(f"No valid shards found in {dataset_root}")

    # 限制shard数量
    if max_shards is not None:
        data_paths = data_paths[:max_shards]
        logger.info(
            f"Using {len(data_paths)} shards out of total {len(find_agibot_shards(dataset_root))} available shards"
        )

    joint_data_list = []
    total_episodes_processed = 0

    for shard_idx, shard_path in enumerate(data_paths):
        logger.info(
            f"Processing shard {shard_idx + 1}/{len(data_paths)}: {shard_path}"
        )

        # LMDB路径
        index_path = os.path.join(shard_path, "index")
        meta_path = os.path.join(shard_path, "meta")

        try:
            # 打开LMDB数据库
            index_env = lmdb.open(index_path, readonly=True, lock=False)
            meta_env = lmdb.open(meta_path, readonly=True, lock=False)

            with index_env.begin() as index_txn:
                # 获取episode数量
                cursor = index_txn.cursor()
                episode_count = 0
                for key, value in cursor:
                    episode_count += 1
                    if (
                        max_episodes_per_shard
                        and episode_count >= max_episodes_per_shard
                    ):
                        break

                episodes_to_process = (
                    min(episode_count, max_episodes_per_shard)
                    if max_episodes_per_shard
                    else episode_count
                )
                logger.info(
                    f"Processing {episodes_to_process} episodes in shard"
                )

                # 重新遍历episodes收集数据
                cursor = index_txn.cursor()
                processed = 0
                shard_samples = 0

                for key, value in tqdm(
                    cursor,
                    desc=f"Shard {shard_idx + 1}",
                    total=episodes_to_process,
                ):
                    if (
                        max_episodes_per_shard
                        and processed >= max_episodes_per_shard
                    ):
                        break

                    try:
                        # 解析episode信息
                        episode_data = pickle.loads(value)
                        uuid = episode_data.get("uuid")
                        if not uuid:
                            continue

                        # 从meta数据库获取joint_state
                        with meta_env.begin() as meta_txn:
                            joint_key = f"{uuid}/observation/robot_state/joint_positions"
                            joint_data_bytes = meta_txn.get(joint_key.encode())

                            if joint_data_bytes is not None:
                                joint_positions = pickle.loads(
                                    joint_data_bytes
                                )
                                joint_positions = np.array(joint_positions)

                                # 处理不同格式的joint数据
                                if len(joint_positions.shape) == 2:
                                    # [time_steps, num_joints] - 采样时间步
                                    num_timesteps = min(
                                        joint_positions.shape[0],
                                        max_timesteps_per_episode,
                                    )
                                    # 均匀采样时间步而不是只取前N个
                                    if num_timesteps > 1:
                                        step_indices = np.linspace(
                                            0,
                                            joint_positions.shape[0] - 1,
                                            num_timesteps,
                                            dtype=int,
                                        )
                                    else:
                                        step_indices = [0]

                                    for step_idx in step_indices:
                                        joint_pos = joint_positions[step_idx]
                                        if len(joint_pos) == 20:
                                            joint_data_list.append(joint_pos)
                                            shard_samples += 1
                                elif len(joint_positions.shape) == 1:
                                    # [num_joints] - 单个时间步
                                    if len(joint_positions) == 20:
                                        joint_data_list.append(joint_positions)
                                        shard_samples += 1

                        processed += 1
                        total_episodes_processed += 1

                    except Exception as e:
                        logger.error(f"Error processing episode {key}: {e}")
                        continue

                logger.info(
                    f"Shard {shard_idx + 1} completed: {processed} episodes, {shard_samples} samples"
                )

            # 关闭数据库
            index_env.close()
            meta_env.close()

        except Exception as e:
            logger.error(f"Error processing shard {shard_path}: {e}")
            continue

    if not joint_data_list:
        raise ValueError("No valid joint data collected!")

    joint_data = np.stack(joint_data_list)  # [N, 20]
    logger.info(
        f"Collected {joint_data.shape[0]} valid joint samples from {total_episodes_processed} episodes across {len(data_paths)} shards"
    )
    return joint_data


def compute_joint_statistics(joint_data):
    """计算joint数据的统计量并生成scale/shift参数

    Args:
        joint_data: [N, 20] numpy array

    Returns:
        statistics: dict with mean, std, min, max, scale_shift parameters
    """
    N, num_joints = joint_data.shape
    logger.info(f"Computing statistics for {N} samples, {num_joints} joints")

    # 基本统计量
    joint_mean = np.mean(joint_data, axis=0)  # [20]
    joint_std = np.std(joint_data, axis=0)  # [20]
    joint_min = np.min(joint_data, axis=0)  # [20]
    joint_max = np.max(joint_data, axis=0)  # [20]
    joint_range = joint_max - joint_min  # [20]

    # 计算scale和shift参数
    # 方法1: 使用标准化 (x - mean) / std
    scale_method1 = joint_std
    shift_method1 = joint_mean

    # 方法2: 使用Min-Max归一化 (x - min) / (max - min)
    # 调整为 (x - shift) / scale 形式
    scale_method2 = joint_range
    shift_method2 = joint_min

    # 方法3: 使用robust统计量 (25%和75%分位数)
    joint_p25 = np.percentile(joint_data, 25, axis=0)
    joint_p75 = np.percentile(joint_data, 75, axis=0)
    joint_iqr = joint_p75 - joint_p25
    scale_method3 = joint_iqr
    shift_method3 = joint_p25

    # 组织结果
    statistics = {
        "num_samples": N,
        "num_joints": num_joints,
        "joint_names": [
            # Left arm joints (7)
            "Joint1_l",
            "Joint2_l",
            "Joint3_l",
            "Joint4_l",
            "Joint5_l",
            "Joint6_l",
            "Joint7_l",
            "left_gripper",  # index 7
            # Right arm joints (7)
            "Joint1_r",
            "Joint2_r",
            "Joint3_r",
            "Joint4_r",
            "Joint5_r",
            "Joint6_r",
            "Joint7_r",
            "right_gripper",  # index 15
            # Head joints (2)
            "joint_head_pitch",
            "joint_head_yaw",
            # Body joints (2)
            "joint_body_pitch",
            "joint_lift_body",
        ],
        "mean": joint_mean,
        "std": joint_std,
        "min": joint_min,
        "max": joint_max,
        "range": joint_range,
        "p25": joint_p25,
        "p75": joint_p75,
        "iqr": joint_iqr,
        "scale_shift_std": [
            [scale_method1[i], shift_method1[i]] for i in range(num_joints)
        ],
        "scale_shift_minmax": [
            [scale_method2[i], shift_method2[i]] for i in range(num_joints)
        ],
        "scale_shift_robust": [
            [scale_method3[i], shift_method3[i]] for i in range(num_joints)
        ],
    }

    return statistics


def print_statistics_summary(stats):
    """打印统计结果摘要"""
    print("\n" + "=" * 80)
    print("AgiBot Joint Statistics Summary")
    print("=" * 80)
    print(f"Total samples: {stats['num_samples']}")
    print(f"Number of joints: {stats['num_joints']}")

    print(
        f"\n{'Joint Name':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Range':<12}"
    )
    print("-" * 80)

    for i, name in enumerate(stats["joint_names"]):
        print(
            f"{name:<20} {stats['mean'][i]:>11.6f} {stats['std'][i]:>11.6f} "
            f"{stats['min'][i]:>11.6f} {stats['max'][i]:>11.6f} {stats['range'][i]:>11.6f}"
        )


def format_scale_shift_for_config(stats, method="std"):
    """格式化scale_shift参数用于config文件

    Args:
        stats: 统计结果字典
        method: 使用的方法 ('std', 'minmax', 'robust')
    """
    method_key = f"scale_shift_{method}"
    if method_key not in stats:
        raise ValueError(f"Method {method} not available")

    scale_shift_list = stats[method_key]

    print("\n" + "=" * 80)
    print(f"Scale-Shift Parameters for Config File (Method: {method})")
    print("=" * 80)
    print(
        "            # AgiBot statistics based on {} samples".format(
            stats["num_samples"]
        )
    )

    # Left arm joints (7 joints)
    print("            # Left arm joints (7 joints)")
    for i in range(7):
        scale, shift = scale_shift_list[i]
        joint_name = stats["joint_names"][i]
        print(f"            [{scale:.9f}, {shift:.9f}],   # {joint_name}")

    # Left gripper
    scale, shift = scale_shift_list[7]
    print(f"            [{scale:.9f}, {shift:.9f}],  # left_gripper")

    # Right arm joints (7 joints)
    print("            # Right arm joints (7 joints)")
    for i in range(8, 15):
        scale, shift = scale_shift_list[i]
        joint_name = stats["joint_names"][i]
        print(f"            [{scale:.9f}, {shift:.9f}],   # {joint_name}")

    # Right gripper
    scale, shift = scale_shift_list[15]
    print(f"            [{scale:.9f}, {shift:.9f}],  # right_gripper")

    # Head joints (2 joints)
    print("            # Head joints (2 joints)")
    for i in range(16, 18):
        scale, shift = scale_shift_list[i]
        print(
            f"            [{scale:.9f}, {shift:.9f}],   # {stats['joint_names'][i]}"
        )

    # Body joints (2 joints)
    print("            # Body joints (2 joints)")
    for i in range(18, 20):
        scale, shift = scale_shift_list[i]
        print(
            f"            [{scale:.9f}, {shift:.9f}],   # {stats['joint_names'][i]}"
        )

    print("=" * 80)


def main():
    # 数据集路径
    dataset_root = "/horizon-bucket/robot_lab2/users/shujie.luo/data/AgiBotWorld-Alpha-250414_34fd7cd4_lmdb"

    # 优化的统计参数 - 目标：统计大量episodes但避免超时
    max_shards = 10  # 使用10个shards
    max_episodes_per_shard = 200  # 每个shard最多200个episodes
    max_timesteps_per_episode = 20  # 每个episode最多20个timesteps

    print("Starting AgiBot joint statistics computation...")
    print(f"Dataset root: {dataset_root}")
    print(f"Max shards: {max_shards}")
    print(f"Max episodes per shard: {max_episodes_per_shard}")
    print(f"Max timesteps per episode: {max_timesteps_per_episode}")
    print(f"Expected episodes: ~{max_shards * max_episodes_per_shard}")
    print(
        f"Expected samples: ~{max_shards * max_episodes_per_shard * max_timesteps_per_episode}"
    )

    try:
        # 1. 收集joint数据
        print("\n1. Collecting joint data...")
        joint_data = collect_joint_data_from_lmdb(
            dataset_root=dataset_root,
            max_shards=max_shards,
            max_episodes_per_shard=max_episodes_per_shard,
            max_timesteps_per_episode=max_timesteps_per_episode,
        )

        # 2. 计算统计量
        print("\n2. Computing statistics...")
        statistics = compute_joint_statistics(joint_data)

        # 3. 保存统计结果
        output_file = "agibot_joint_statistics_large.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(statistics, f)
        print(f"\n3. Statistics saved to {output_file}")

        # 4. 打印结果
        print_statistics_summary(statistics)

        # 5. 生成推荐的config参数
        print("\n" + "=" * 80)
        print("RECOMMENDED: Using standard deviation method for scale/shift")
        format_scale_shift_for_config(statistics, method="std")

        print(
            f"\nDone! Statistics computed from {statistics['num_samples']} samples."
        )
        print("Copy the scale_shift configuration to your config file.")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
