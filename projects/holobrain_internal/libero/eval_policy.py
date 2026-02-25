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

_original_torch_load = torch.load


def _safe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _safe_torch_load


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)


def run_task(
    usr_args, task_id, task_suite, log_file, total_episodes, total_successes
):  # noqa: E501
    task_suite_name = usr_args["task_suite"]
    num_trials_per_task = usr_args["num_trials_per_task"]
    num_steps_wait = usr_args["num_steps_wait"]
    save_video = usr_args["save_video"]
    policy_name = usr_args["policy_name"]
    get_model = eval_function_decorator(policy_name, "get_model")
    model = get_model(usr_args)
    eval_func = eval_function_decorator(policy_name, "predict_action")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO environment and task description
    env, task_description = get_libero_env(task, resolution=256)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
        print(f"\nTask: {task_description}")
        log_message(f"\nTask: {task_description}\n", log_file)

        # Reset environment
        env.reset()
        reset_func(model)
        # Set initial states
        obs = env.set_init_state(initial_states[episode_idx])

        # Setup
        t = 0
        replay_images = []
        if task_suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif task_suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif task_suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif task_suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps

        log_message(f"Starting episode {task_episodes + 1}...\n", log_file)
        action_counter = 0
        while t < max_steps + num_steps_wait:
            # try:
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
            # wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
            wrist_extrinsic = get_camera_extrinsic_matrix(
                env.sim, "robot0_eye_in_hand"
            )
            wrist_intrinsic = get_camera_intrinsic_matrix(
                env.sim,
                "robot0_eye_in_hand",
                camera_height=256,
                camera_width=256,
            )
            # img = np.stack([side_img, wrist_img])
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
                # "robot_state": np.concatenate(
                #         (
                #           gripper_state[None],
                #           obs["robot0_eef_pos"],
                #           np.roll(obs["robot0_eef_quat"],
                #           1))
                #     )[None].astype(np.float32),
                "robot_state": robot_state.astype(np.float32),
            }
            if action_counter == 0:
                action = eval_func(model, observation, task_description)
                action_counter = action.shape[0]

            for robot in env.env.robots:
                robot.controller.use_delta = False

            step_action = action[-action_counter]
            step_action_quat = T.quat2axisangle(step_action[4:8][[1, 2, 3, 0]])  # noqa: E501
            step_arm_action = np.concatenate(
                (step_action[1:4], step_action_quat)
            )
            step_action = np.concatenate((step_arm_action, step_action[:1]))
            obs, reward, done, info = env.step(step_action.tolist())
            action_counter -= 1
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1
            # except Exception as e:
            #     log_message(f"Caught exception: {e}", log_file)
            #     break

        task_episodes += 1
        total_episodes += 1

        if save_video:
            # Save a replay video of the episode
            save_rollout_video(
                f"{usr_args['output_dir']}/task_{task_id}",
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=None,
            )

        # Log current results
        log_message(f"Success: {done}\n", log_file)
        log_message(
            f"# episodes completed so far: {total_episodes}\n", log_file
        )  # noqa: E501
        log_message(
            f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n",  # noqa: E501
            log_file,
        )  # noqa: E501

    # Log final results
    log_message("\n" + "=" * 25 + "TASK RESULTS" + "=" * 25, log_file)  # noqa: E501
    current_task_success_rate = (
        float(task_successes) / float(task_episodes)
        if task_episodes > 0
        else 0.0
    )
    log_message(
        f"Current task success rate: {current_task_success_rate}\n",  # noqa: E501
        log_file,
    )
    log_message(
        f"Current total success rate: {float(total_successes) / float(total_episodes)}\n",  # noqa: E501
        log_file,
    )

    try:
        env.close()
    except Exception:
        pass

    return total_episodes, total_successes, current_task_success_rate


def main(usr_args):
    set_seed_everywhere(seed=7)
    task_suite_name = usr_args["task_suite"]
    os.makedirs(usr_args["output_dir"], exist_ok=True)
    local_log_filepath = os.path.join(usr_args["output_dir"], "log.txt")
    log_file = open(local_log_filepath, "w", encoding="utf-8")
    print(f"Logging to local log file: {local_log_filepath}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {task_suite_name}")
    log_message(f"Task suite: {task_suite_name}\n", log_file)
    if usr_args["task_id"] != -1:
        task_range = usr_args["task_id"]
        print(f"Running specific task ID: {usr_args['task_id']}")
    else:
        task_range = range(num_tasks_in_suite)
    total_episodes, total_successes = 0, 0
    per_task_results = {}
    for task_id in tqdm.tqdm(task_range):
        print("task_id:", task_id)
        total_episodes, total_successes, task_success_rate = run_task(
            usr_args,
            task_id,
            task_suite,
            log_file,
            total_episodes,
            total_successes,
        )
        per_task_results[f"task_{task_id}"] = task_success_rate
    # Calculate final success rate
    final_success_rate = (
        float(total_successes) / float(total_episodes)
        if total_episodes > 0
        else 0
    )

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(
        f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)",  # noqa: E501
        log_file,
    )

    json_results = {task_suite_name: per_task_results}

    json_path = os.path.join(usr_args["output_dir"], "results.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=4)
        log_message(f"Results saved to {json_path}", log_file)
    except Exception as e:
        log_message(f"Failed to save JSON results: {e}", log_file)

    if log_file:
        log_file.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--model_prefix", type=str, default="model_0")
    parser.add_argument(
        "--vlm_ckpt_dir",
        type=str,
        default="/horizon-bucket/robot_lab/users/xuewu.lin/ckpt",
    )
    parser.add_argument(
        "--urdf_dir",
        type=str,
        default="/horizon-bucket/robot_lab/users/xuewu.lin/urdf",
    )
    parser.add_argument(
        "--model_processor", type=str, default="libero_processor"
    )
    parser.add_argument("--task_suite", type=str, default="libero_goal")
    parser.add_argument(
        "--task_id", default="-1", help="Specific task id to run"
    )  # noqa: E501
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--save_video", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="eval_result")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")  # noqa: B005
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:  # noqa: E722
                value = value
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)
    config["output_dir"] = args.output_dir
    config["task_id"] = json.loads(args.task_id)
    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    print(usr_args)
    main(usr_args)
