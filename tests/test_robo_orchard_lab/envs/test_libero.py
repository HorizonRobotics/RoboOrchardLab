# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

import pytest
from robo_orchard_core.datatypes import (
    BatchCameraData,
    BatchFrameTransform,
    BatchJointsState,
)

try:
    import libero  # noqa: F401
except ImportError:
    pytest.skip("libero is not installed", allow_module_level=True)

import torch
from robo_orchard_core.utils.math import (
    matrix_to_quaternion,
)
from robosuite.controllers.osc import OperationalSpaceController

from robo_orchard_lab.envs.libero import LiberoEvalEnv, LiberoEvalEnvCfg


@pytest.fixture()
def dummy_libero_eval_env_cfg() -> LiberoEvalEnvCfg:
    return LiberoEvalEnvCfg(
        suite_name="libero_10",
        task_id=0,
        episode_idx=0,
        camera_depths=True,
    )


@pytest.fixture()
def dummy_libero_eval_env(
    dummy_libero_eval_env_cfg: LiberoEvalEnvCfg,
):
    env = dummy_libero_eval_env_cfg()
    yield env
    env.close()


@pytest.fixture()
def dummy_libero_eval_env_formatted():
    cfg = LiberoEvalEnvCfg(
        suite_name="libero_10",
        task_id=0,
        episode_idx=0,
        camera_depths=True,
        format_datatypes=True,
    )
    env = cfg()
    yield env
    env.close()


class TestLiberoEvalEnv:
    def test_libero_eval_env_step(
        self, dummy_libero_eval_env: LiberoEvalEnv
    ) -> None:
        env = dummy_libero_eval_env
        obs, info = env.reset()
        assert "agentview_depth" in obs
        assert info is not None
        action = torch.zeros((7,), dtype=torch.float32)
        step_return = env.step(action)
        assert step_return.observations is not None
        print("reward: ", step_return.rewards)
        print(
            "terminated: ",
            step_return.terminated,
            " type: ",
            type(step_return.terminated),
        )
        print("truncated: ", step_return.truncated)
        print(env.task_info)
        step_time_interval = env.control_timestep
        print("control_timestep: ", step_time_interval)
        print("obs fps: ", 1.0 / step_time_interval)
        assert step_return.truncated is False
        assert step_return.terminated is False

    def test_step_limit(self, dummy_libero_eval_env: LiberoEvalEnv) -> None:
        env = dummy_libero_eval_env
        env.reset()
        action = torch.zeros((7,), dtype=torch.float32)

        for step_idx in range(env.step_limit):
            ret = env.step(action)
            if step_idx + 1 >= env.step_limit:
                assert ret.truncated is True
            else:
                assert ret.truncated is False

    def test_format_datatypes(
        self, dummy_libero_eval_env_formatted: LiberoEvalEnv
    ) -> None:
        env = dummy_libero_eval_env_formatted
        obs, info = env.reset()
        print("obs keys: ", obs.keys())
        assert "agentview_image" in obs
        agentview_depth = obs["agentview_depth"]
        assert isinstance(agentview_depth, BatchCameraData)
        joints = obs["joints"]
        assert isinstance(joints, BatchJointsState)
        print("joints: ", joints)
        assert joints.timestamps is not None
        tf_world = obs["tf_world"]
        print("all frame ids: ", tf_world.keys())
        assert isinstance(tf_world, dict)
        for frame_id, tf in tf_world.items():
            assert isinstance(tf, BatchFrameTransform)
            assert tf.child_frame_id == frame_id
            assert tf.timestamps is not None
            assert tf.timestamps == joints.timestamps

    def test_get_last_action(
        self, dummy_libero_eval_env_formatted: LiberoEvalEnv
    ):
        env = dummy_libero_eval_env_formatted
        obs, info = env.reset()
        last_action = env.get_last_action()
        assert last_action is None

        action = torch.tensor(
            [0.4, 0.1, -0.2, 0.1, -0.1, 0.2, -1], dtype=torch.float32
        )
        _ = env.step(action)
        last_action = env.get_last_action()
        assert last_action is not None
        assert isinstance(last_action.goal_eef, BatchFrameTransform)
        assert isinstance(last_action.osc_arm_action, torch.Tensor)
        assert isinstance(last_action.osc_gripper_action, torch.Tensor)

        recovered_action = env.get_arm_osc_delta_pose_from_target_pose(
            target_pose=last_action.goal_eef,
            source_pose=obs["tf_world"][env.eef_name],
        )
        print("recovered_action: ", recovered_action)
        print("last_action.osc_arm_action: ", last_action.osc_arm_action)

        print("last_action: ", last_action)
        print("last eef: ", obs["tf_world"][env.eef_name])

        # Verify that the recovered action matches the original action
        # Important: here use atol=1e-3 because in robosuite, the controller
        # is applied after one physic step, which introduce small error.
        # The observation is retrieved before the physics step.
        assert torch.allclose(
            recovered_action,
            last_action.osc_arm_action.to(recovered_action),
            atol=4e-3,
        )

    def test_eef_name_consistent_with_controller(
        self, dummy_libero_eval_env_formatted: LiberoEvalEnv
    ):
        env = dummy_libero_eval_env_formatted
        obs, info = env.reset()
        action = torch.tensor(
            [0.4, 0.1, -0.2, 0.1, -0.1, 0.2, -1], dtype=torch.float32
        )
        step_ret = env.step(action)
        for _ in range(10):
            act = [0.0] * 7
            act[-1] = -1
            step_ret = env.step(act)
        eef: BatchFrameTransform = step_ret.observations["tf_world"][
            env.eef_name
        ]

        controller: OperationalSpaceController = env._env.robots[0].controller
        # test if the eef name is consistent with controller
        assert torch.allclose(
            eef.xyz[0], torch.from_numpy(controller.ee_pos), atol=1e-4
        )
        ee_quat = matrix_to_quaternion(
            torch.from_numpy(controller.ee_ori_mat.reshape(1, 3, 3))  # type: ignore # noqa
        )
        assert torch.allclose(eef.quat, ee_quat, atol=1e-4)

    def test_camera_depth_with_tf(
        self, dummy_libero_eval_env_formatted: LiberoEvalEnv
    ):
        env = dummy_libero_eval_env_formatted
        obs, info = env.reset()
        action = torch.tensor(
            [0.4, 0.1, -0.2, 0.1, -0.1, 0.2, -1], dtype=torch.float32
        )
        step_ret = env.step(action)
        depth_img: BatchCameraData = step_ret.observations["agentview_depth"]
        camera_img: BatchCameraData = step_ret.observations["agentview_image"]

        for obj_keys in ["cream_cheese_1_pos", "butter_1_pos"]:
            points_3d = torch.from_numpy(
                step_ret.observations[obj_keys].reshape(1, 3).copy()
            )
            # project the 3d points to 2d pixel coordinates
            proj_p = camera_img.project_points_to_image(
                points_3d, frame_id="world"
            )
            point_xy = torch.round(proj_p[0, 0, :2]).to(torch.int)
            point_depth = proj_p[0, 0, 2]
            print("projected pixel: ", point_xy)
            print("point depth from projection: ", point_depth)
            # get the depth value from depth image
            depth_value = depth_img.sensor_data[0, point_xy[1], point_xy[0], 0]
            print("depth value from depth image: ", depth_value)
            # verify the depth value, the error should be smaller
            # than 5cm because the objects are small
            assert torch.isclose(
                point_depth.to(depth_value), depth_value, atol=5e-2
            )
