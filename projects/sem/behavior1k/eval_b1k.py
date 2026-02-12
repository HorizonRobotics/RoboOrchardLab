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


import csv
import json
import logging
import os
import sys
import traceback
from inspect import getsourcefile
from pathlib import Path
from signal import SIGINT, signal
from typing import Any, List, Tuple

import cv2
import hydra
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T  # noqa: N812
import torch as th
from av.container import Container
from av.stream import Stream
from deploy_policy import SEMPolicy
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.array_tensor_utils import torch_to_numpy
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.learning.utils.obs_utils import (
    create_video_writer,
    write_video,
)
from omnigibson.macros import create_module_macros, gm, macros
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 10

# set global variables to boost performance
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True

# Set grasp window to larger value to account for hard grasps
with macros.unlocked():
    # macros.robots.manipulation_robot.GRASP_WINDOW = 0.75
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.1


# create module logger
logger = logging.getLogger("evaluator")
logger.setLevel(20)  # info


def quat_to_yaw(quat):
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return yaw


def yaw_to_quat(yaw):
    half = yaw * 0.5
    q = np.zeros((*yaw.shape, 4), dtype=np.float32)
    q[..., 2] = np.sin(half)  # z
    q[..., 3] = np.cos(half)  # w
    return q


def traj_local_to_world(
    traj_local,
    xyz0_world,
    quat0_world,
):
    """traj_local_to_world.

    Args:
        traj_local: (T, 3) -> [x_local, y_local, yaw_local]
        xyz0_world: (3,)  -> anchor world position
        quat0_world: (4,) -> anchor world quaternion (x, y, z, w)

    Returns:
        xyz_world: (T, 3)
        quat_world: (T, 4)
    """
    traj_local = np.asarray(traj_local)
    xyz0_world = np.asarray(xyz0_world)
    quat0_world = np.asarray(quat0_world)

    xy_local = traj_local[:, :2]  # (T, 2)
    yaw_local = traj_local[:, 2]  # (T,)

    yaw0 = quat_to_yaw(quat0_world)

    # rotation matrix local -> world
    c, s = np.cos(yaw0), np.sin(yaw0)
    rot = np.array(
        [
            [c, -s],
            [s, c],
        ]
    )  # noqa: N812

    # position
    xy_world = (rot @ xy_local.T).T + xyz0_world[:2]
    z_world = np.full((len(traj_local), 1), xyz0_world[2])
    xyz_world = np.concatenate([xy_world, z_world], axis=1)

    # orientation
    yaw_world = yaw_local + yaw0
    yaw_world = (yaw_world + np.pi) % (2 * np.pi) - np.pi
    quat_world = yaw_to_quat(yaw_world)

    return xyz_world, quat_world


class Evaluator:
    """Evaluator for behavior1k.

    Evaluator class for running and evaluating policies for behavior task.
    This class manages the setup, execution,
    and evaluation of policy rollouts in OmniGibson environment,
    tracking metrics such as the number of trials, successes, and total time.
    It supports loading environments, robots, policies, and metrics,
    and provides methods for stepping through the environment, resetting state,
    and handling video outputs and loggings.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # record total number and success number of trials and trial time
        self.n_trials = 0
        self.n_success_trials = 0
        self.total_time = 0
        self.robot_action = dict()

        self.policy = self.load_policy()
        self.env = self.load_env(env_wrapper=self.cfg.env_wrapper)
        self.robot = self.load_robot()
        self.metrics = self.load_metrics()

        self.reset()

        # manually reset environment episode number
        self.env._current_episode = 0
        self._video_writer = None

    def load_env(self, env_wrapper: DictConfig) -> EnvironmentWrapper:
        """Load evn.

        Read the environment config file and create the environment.
        The config file is located in the configs/envs directory.
        """
        # Disable a subset of transition rules for data collection
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        # Load config file
        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, (
            f"Got invalid task name: {task_name}"
        )

        # Now, get human stats of the task
        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }

        meta_file_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-task-instances",
            "metadata",
            "episodes.jsonl",
        )
        with open(meta_file_path, "r") as f:
            episodes = [json.loads(line) for line in f]

        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])

        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(
                self.human_stats[k]
            )

        # Load the seed instance by default
        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", (
            f"Got invalid robot type: {robot_type}, only R1Pro is supported."
        )

        cfg = generate_basic_environment_config(
            task_name=task_name, task_cfg=task_cfg
        )

        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(
                activity_name=task_name
            )
            relevant_rooms = augment_rooms(
                relevant_rooms, task_cfg["scene_model"], task_name
            )
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(
                task_name=task_name,
                task_cfg=task_cfg,
            )
        ]

        # Update observation modalities
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(
            PROPRIOCEPTION_INDICES["R1Pro"].keys()
        )

        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(
                self.cfg.robot.controllers
            )

        if self.cfg.max_steps is None:
            logger.info(
                "Setting timeout to be 2x the average length of human demos: "
                "{int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(
                self.human_stats["length"] * 2
            )
        else:
            logger.info(
                f"Setting timeout to be "
                f"{self.cfg.max_steps} steps through config"
            )
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps

        cfg["task"]["include_obs"] = False

        env = og.Environment(configs=cfg)

        # instantiate env wrapper
        env = instantiate(env_wrapper, env=env)
        return env

    def load_robot(self) -> BaseRobot:
        """Loads and returns the robot instance from the environment.

        Returns:
            BaseRobot: The robot instance loaded from the environment.
        """
        robot = self.env.scene.object_registry("name", "robot_r1")
        with og.sim.stopped():
            robot.base_footprint_link.mass = 250.0

        return robot

    def load_policy(self) -> Any:
        """Loads and returns the policy instance."""
        policy = SEMPolicy(
            model_path=self.cfg["model_path"],
            processor=self.cfg["model_processor"],
            model_prefix=self.cfg["model_prefix"],
        )
        return policy

    def load_metrics(self) -> List[MetricBase]:
        """Load agent and task metrics."""
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]

    def step(self, pos, quat, action) -> Tuple[bool, bool]:
        """Performs a single step of the task by executing the policy.

        interacting with the environment, processing observations,
        and updating metrics, tracking trial success.

        Returns:
            Tuple[bool, bool]:
                - terminated (bool): Whether the episode has terminated
                    (i.e., reached a terminal state)
                - truncated (bool): Whether the episode was truncated
                    (i.e., stopped due to a time limit or other constraint)

        Workflow:
            1. Computes the next action using the policy
                based on the current observation.
            2. Steps the environment with the computed action
                and retrieves the next observation, termination
                and truncation flags, and additional info.
            3. If the episode has ended (terminated or truncated),
                increments the trial counter
                and updates the count of successful trials
                if the task was completed successfully.
            4. Preprocesses the new observation.
            5. Invokes step callbacks for all registered metrics
                to update their state.
            6. Returns the termination and truncation status.
        """

        self.robot.set_position_orientation(pos, quat)
        obs, _, terminated, truncated, info = self.env.step(
            action, n_render_iterations=1
        )

        # process obs
        self.obs = self._preprocess_obs(obs)

        if terminated or truncated:
            self.n_trials += 1
            if info["done"]["success"]:
                self.n_success_trials += 1

        for metric in self.metrics:
            metric.step_callback(self.env)

        return terminated, truncated

    @property
    def video_writer(self) -> Tuple[Container, Stream]:
        """Returns the video writer for the current evaluation step."""
        return self._video_writer

    @video_writer.setter
    def video_writer(self, video_writer: Tuple[Container, Stream]) -> None:
        if self._video_writer is not None:
            (container, stream) = self._video_writer

            # Flush any remaining packets
            for packet in stream.encode():
                container.mux(packet)

            # Close the container
            container.close()
        self._video_writer = video_writer

    def load_task_instance(
        self, instance_id: int, test_hidden: bool = False
    ) -> None:
        """Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only]
                Whether to load the hidden test instance.
        """
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                get_task_instance_path(scene_model),
                f"json/"
                f"{scene_model}_task_{self.env.task.activity_name}_instances/"
                f"{tro_filename}-tro_state.json",
            )

        with open(tro_file_path, "r") as f:
            tro_states = recursively_convert_to_torch(json.load(f))

        for tro_key, tro_state in tro_states.items():
            if tro_key == "robot_poses":
                # presampled_robot_poses = tro_state
                robot_pos = tro_state[self.robot.model_name][0]["position"]
                robot_quat = tro_state[self.robot.model_name][0]["orientation"]

                self.robot.set_position_orientation(robot_pos, robot_quat)

                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(
                    tro_state, serialized=False
                )

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance,
        # but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        """Preprocess the observation before passing it to the policy.

        Args:
            obs (dict): The observation dictionary to preprocess.

        Returns:
            dict: The preprocessed observation dictionary.
        """
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        # First camera query may be zero.
        # get_position_orientation() returns latest pose.
        # OG render is async; pose may be ahead.
        # It needs >=3 renders to sync.
        # We use one render for speed.
        # So we use synced camera parameters.
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(
                        T.relative_pose_transform(
                            *(camera.get_position_orientation()), *base_pose
                        )
                    )
                )
            else:
                cam_pose = T.mat2pose(
                    th.tensor(
                        np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T),
                        dtype=th.float32,
                    )
                )
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*cam_pose, *base_pose))
                )

        obs["robot_r1::cam_rel_poses"] = th.cat(cam_rel_poses, axis=-1)

        # append task id to obs
        obs["task_id"] = th.tensor(
            [TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64
        )

        obs["instruction"] = self.cfg.instruction
        obs = torch_to_numpy(obs)

        return obs

    def _write_video(self) -> None:
        """Write the current robot observations to video."""
        if ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb" not in self.obs:
            return

        # concatenate obs
        left_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"],
            (224, 224),
        )
        right_wrist_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"],
            (224, 224),
        )
        head_rgb = cv2.resize(
            self.obs[ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"],
            (448, 448),
        )
        write_video(
            np.expand_dims(
                np.hstack(
                    [np.vstack([left_wrist_rgb, right_wrist_rgb]), head_rgb]
                ),
                0,
            ),
            video_writer=self.video_writer,
            batch_size=1,
            mode="rgb",
        )

    def reset(self) -> None:
        """Reset the environment, policy, and compute metrics."""
        self.obs = self._preprocess_obs(self.env.reset()[0])

        # run metric start callbacks
        for metric in self.metrics:
            metric.start_callback(self.env)

        self.policy.reset()

        self.n_success_trials, self.n_trials = 0, 0

    def __enter__(self):
        signal(SIGINT, self._sigint_handler)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"Total success trials: {self.n_success_trials}")
        logger.info(f"Total trials: {self.n_trials}")

        if self.n_trials > 0:
            logger.info(
                f"Success rate: {self.n_success_trials / self.n_trials}"
            )

        logger.info("=" * 50)
        logger.info("")

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, exc_tb)

        self.video_writer = None
        self.env.close()
        og.shutdown()

    def _sigint_handler(self, signal_received, frame):
        logger.warning("SIGINT or CTRL-C detected.\n")
        self.__exit__(None, None, None)
        sys.exit(0)


if __name__ == "__main__":
    register_omegaconf_resolvers()

    config_dir = Path(getsourcefile(lambda: 0)).parents[0] / "configs"
    with hydra.initialize_config_dir(
        config_dir=str(config_dir),
        version_base="1.1",
    ):
        config = hydra.compose(
            config_name="base_config.yaml",
            overrides=sys.argv[1:],
        )

        # load task config
        task_cfg = OmegaConf.load(config_dir / "tasks.yaml")

    OmegaConf.resolve(config)
    OmegaConf.resolve(task_cfg)
    # config.instruction = task_cfg[config.task.name]
    with open_dict(config):
        config.instruction = task_cfg[config.task.name]

    # set headless mode
    gm.HEADLESS = config.headless

    # set video path
    if config.write_video:
        video_path = Path(config.log_path).expanduser() / "videos"
        video_path.mkdir(parents=True, exist_ok=True)

    assert not (config.eval_on_train_instances and config.test_hidden), (
        "Cannot eval on train instances "
        "and test hidden instances simultaneously."
    )

    if config.test_hidden:
        logger.info(
            "You are evaluating on hidden test instances! "
            "This is for internal use only."
        )

    # get run instances
    if config.eval_on_train_instances:
        logger.info(
            "You are evaluating on training instances, "
            "set eval_on_train_instances to False for test instances."
        )

        task_idx = TASK_NAMES_TO_INDICES[config.task.name]
        meta_file_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-task-instances",
            "metadata",
            "episodes.jsonl",
        )
        with open(meta_file_path, "r") as f:
            episodes = [json.loads(line) for line in f]

        instances_to_run = []
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                instances_to_run.append(
                    str(int((episode["episode_index"] // 10) % 1e3))
                )

        if config.eval_instance_ids:
            assert set(config.eval_instance_ids).issubset(
                set(range(m.NUM_TRAIN_INSTANCES))
            ), f"eval instance ids must be in range({m.NUM_TRAIN_INSTANCES})"

            instances_to_run = [
                instances_to_run[i] for i in config.eval_instance_ids
            ]

    elif config.test_hidden:
        instances_to_run = (
            config.eval_instance_ids
            if config.eval_instance_ids is not None
            else set(range(m.NUM_EVAL_INSTANCES))
        )
        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"
    else:
        instances_to_run = OmegaConf.select(
            config, "instances_to_run", default=None
        )

        if instances_to_run is None:
            instances_to_run = (
                config.eval_instance_ids
                if config.eval_instance_ids is not None
                else set(range(m.NUM_EVAL_INSTANCES))
            )

        assert set(instances_to_run).issubset(
            set(range(m.NUM_EVAL_INSTANCES))
        ), f"eval instance ids must be in range({m.NUM_EVAL_INSTANCES})"

        # load csv file
        task_instance_csv_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-task-instances",
            "metadata",
            "test_instances.csv",
        )
        with open(task_instance_csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]

        assert (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][1]
            == config.task.name
        ), (
            f"Task name from config {config.task.name} "
            f"does not match task name from csv "
            f"{lines[TASK_NAMES_TO_INDICES[config.task.name]][1]}"
        )

        test_instances = (
            lines[TASK_NAMES_TO_INDICES[config.task.name]][2]
            .strip()
            .split(",")
        )
        instances_to_run = [int(test_instances[i]) for i in instances_to_run]

    # establish metrics
    metrics = {}
    metrics_path = Path(config.log_path).expanduser() / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with Evaluator(config) as evaluator:
        logger.info("Starting evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for evaluation...")
            for epi in range(m.NUM_EVAL_EPISODES):
                evaluator.reset()
                done = False
                if config.write_video:
                    video_name = (
                        str(video_path)
                        + f"/{config.task.name}_{idx}_{epi}.mp4"
                    )
                    evaluator.video_writer = create_video_writer(
                        fpath=video_name,
                        resolution=(448, 672),
                    )
                # run metric start callbacks
                for metric in evaluator.metrics:
                    metric.start_callback(evaluator.env)

                while not done:
                    action, traj = evaluator.policy.forward(obs=evaluator.obs)

                    zeros = np.zeros(
                        (action.shape[0], 3),
                        dtype=action.dtype,
                    )
                    action = np.concatenate([zeros, action], axis=1)

                    base_pos, base_quat = (
                        evaluator.robot.get_position_orientation()
                    )
                    pos, quat = traj_local_to_world(traj, base_pos, base_quat)

                    for i in range(action.shape[0] // 2):
                        terminated, truncated = evaluator.step(
                            pos[i],
                            quat[i],
                            action[i, :],
                        )
                        if terminated or truncated:
                            done = True
                            break

                        if config.write_video:
                            evaluator._write_video()

                        if evaluator.env._current_step % 1000 == 0:
                            logger.info(
                                f"Current step: {evaluator.env._current_step}"
                            )

                # run metric end callbacks
                for metric in evaluator.metrics:
                    metric.end_callback(evaluator.env)

                logger.info(
                    f"Evaluation finished at step "
                    f"{evaluator.env._current_step}."
                )
                logger.info(
                    f"Evaluation exit state: {terminated}, {truncated}"
                )
                logger.info(f"Total trials: {evaluator.n_trials}")
                logger.info(
                    f"Total success trials: {evaluator.n_success_trials}"
                )

                # gather metric results and write to file
                for metric in evaluator.metrics:
                    metrics.update(metric.gather_results())

                with open(
                    metrics_path / f"{config.task.name}_{idx}_{epi}.json", "w"
                ) as f:
                    json.dump(metrics, f)

                # reset video writer
                if config.write_video:
                    evaluator.video_writer = None
                    logger.info(f"Saved video to {video_name}")
                else:
                    logger.warning("No observations were recorded.")
