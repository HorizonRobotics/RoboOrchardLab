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

import argparse
import importlib
import json
import logging
import os

import numpy as np
import robosuite.utils.transform_utils as T  # noqa: N812
import torch
import tqdm
import yaml
from libero.libero import benchmark
from libero_utils import (
    get_libero_agentview_image,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_wrist_image,
    save_rollout_video,
    set_seed_everywhere,
)
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)

from robo_orchard_lab.dataset.libero.utils import transform_ee_rotations
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)


MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}

torch.serialization.add_safe_globals(
    [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Float64DType,
    ]
)


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def run_task(args, config, task_id, task_suite):
    task_suite_name = args.task_suite
    num_trials_per_task = args.num_trials_per_task
    num_steps_wait = args.num_steps_wait
    save_video = args.save_video
    policy_name = args.policy_name
    get_model = eval_function_decorator(policy_name, "get_model")
    model = get_model(config)
    eval_func = eval_function_decorator(policy_name, "predict_action")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO environment and task description
    env, task_description = get_libero_env(task, resolution=256)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
        logger.info(f"Task: {task_description}")

        # Reset environment
        env.reset()
        reset_func(model)
        # Set initial states
        obs = env.set_init_state(initial_states[episode_idx])

        # Setup
        t = 0
        replay_images = []
        max_steps = MAX_STEPS.get(task_suite_name, 300)

        logger.info(f"Starting episode {task_episodes + 1}...")
        action_counter = 0
        while t < max_steps + num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects # noqa: E501
            # and we need to wait for them to fall
            if t < num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action())
                t += 1
                continue
            side_img, side_img_depth = get_libero_agentview_image(env, obs)
            side_extrinsic = get_camera_extrinsic_matrix(env.sim, "agentview")
            side_intrinsic = get_camera_intrinsic_matrix(
                env.sim, "agentview", camera_height=256, camera_width=256
            )
            wrist_img, wrist_img_depth = get_libero_wrist_image(env, obs)
            wrist_extrinsic = get_camera_extrinsic_matrix(
                env.sim, "robot0_eye_in_hand"
            )
            wrist_intrinsic = get_camera_intrinsic_matrix(
                env.sim,
                "robot0_eye_in_hand",
                camera_height=256,
                camera_width=256,
            )
            replay_images.append(side_img)
            current_width = (
                obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
            )
            MAX_WIDTH = 0.08  # noqa: N806
            gripper_state = current_width / MAX_WIDTH

            r_diff = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            ee_state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    T.quat2axisangle(obs["robot0_eef_quat"]),
                )
            )[None].astype(np.float32)
            ee_quat = transform_ee_rotations(ee_state, r_diff)
            ee_state = np.concatenate([ee_state[:, :3], ee_quat], axis=1)
            robot_state = np.concatenate(
                [gripper_state[None, None], ee_state], axis=1
            )
            observation = {
                "observation": {
                    "agentview": {
                        "rgb": side_img,
                        "depth": side_img_depth,
                        "extrinsic_cv": side_extrinsic,
                        "intrinsic_cv": side_intrinsic,
                    },
                    "eye_in_hand": {
                        "rgb": wrist_img,
                        "depth": wrist_img_depth,
                        "extrinsic_cv": wrist_extrinsic,
                        "intrinsic_cv": wrist_intrinsic,
                    },
                },
                "robot_state": robot_state.astype(np.float32),
            }
            if action_counter == 0:
                action = eval_func(model, observation, task_description)
                action_counter = action.shape[0]

            for robot in env.env.robots:
                robot.controller.use_delta = False

            step_action = action[-action_counter]
            step_action_quat = T.quat2axisangle(step_action[4:8][[1, 2, 3, 0]])
            step_arm_action = np.concatenate(
                (step_action[1:4], step_action_quat)
            )
            step_action = np.concatenate((step_arm_action, step_action[:1]))
            obs, _, done, _ = env.step(step_action.tolist())
            action_counter -= 1
            if done:
                task_successes += 1
                break
            t += 1
        task_episodes += 1

        if save_video:
            # Save a replay video of the episode
            save_rollout_video(
                args.output_dir,
                replay_images,
                task_episodes,
                success=done,
                task_description=task_description,
                log_file=None,
            )
        logger.info(
            f"Success: {done}, success rate: {task_successes}/{task_episodes}"
        )

    logger.info("\n" + "=" * 25 + "TASK RESULTS" + "=" * 25)
    logger.info(
        f"{task_suite_name}_task_{task_id} "
        f"success rate: {task_successes}/{task_episodes}"
    )

    try:
        env.close()
    except Exception:
        pass

    return task_successes, task_episodes


def main(args, config):
    set_seed_everywhere(seed=7)
    task_suite_name = args.task_suite
    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_id = args.task_id
    assert 0 <= task_id < num_tasks_in_suite, (
        f"task_id {task_id} is out of range [0, {num_tasks_in_suite})"
    )
    logger.info(f"Task suite: {task_suite_name}, task_id: {task_id}")

    task_successes, task_episodes = run_task(
        args,
        config,
        task_id,
        task_suite,
    )
    json_results = {
        "success_rate": task_successes / task_episodes
        if task_episodes > 0
        else 0
    }
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4)


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task_suite", type=str, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--save_video", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="eval_result")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i][2:] if pairs[i].startswith("--") else pairs[i]
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception:
                value = value
            if value is not None:
                override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)
    args.policy_name = config.get("policy_name", "default_policy")
    return args, config


if __name__ == "__main__":
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    args, config = parse_args_and_config()
    main(args, config)
