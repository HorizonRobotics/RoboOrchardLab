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


import asyncio
import dataclasses
import functools
import http
import logging
import os
import socket
import time
import traceback

import msgpack
import numpy as np
import torch
import tyro
import websockets
import websockets.sync.client
import yaml
from robo_orchard_core.utils.config import load_config_class

from robo_orchard_lab.dataset.behavior import utils
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.models.sem_modules.processor import (
    MultiArmManipulationInput,
)
from robo_orchard_lab.utils.path import in_cwd

try:
    import websockets.asyncio.server as _server
except ImportError:
    # Fallback for websockets < 13.0
    print("import websockets.server")
    import websockets.server as _server

from copy import deepcopy
from typing import Any, Optional

DEBUG = False
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        else:
            # For older websockets versions, return a simple response
            return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    # Continue with the normal request handling.
    return None


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(
            f"Starting websocket server on {self._host}:{self._port}..."
        )
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv())
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                action = {
                    "action": action.cpu().numpy(),
                }
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time
                    # since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = (
                        prev_total_time * 1000
                    )

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(
                    f"Connection from {websocket.remote_address} closed"
                )
                break
            except Exception:
                logger.error(
                    f"Error in connection from {websocket.remote_address}:\n"
                    f"{traceback.format_exc()}"
                )
                if DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    # Try new websockets API first
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                except AttributeError:
                    # Fallback for older websockets versions
                    await websocket.close(
                        code=1011, reason="Internal server error"
                    )
                raise


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # If provided, will be used in case the "prompt" key is not present
    # in the data, or if the model doesn't have a default

    # prompt.
    default_prompt: str | None = None

    # Dataset root, used to retrieve the prompt of the task
    # if taskname is not None.
    dataset_root: str | None = "/work/bucket/2025-challenge-demos"
    # If provided, will be used to retrieve the prompt of the task,
    # otherwise use turning_on_radio as default.
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8001

    # Specifies how to load the policy. If not provided,
    # the default policy for the environment will be used.
    policy_config: str | None = "behavior_policy/deploy_policy.yml"


class SEMPolicy:
    def __init__(
        self, config, processor=None, vlm_ckpt_dir=None, urdf_dir=None
    ):
        if processor is None:
            processor = "processor"
        # logger.info(f"model config: {config}, processor: {processor}")

        target_vlm_ckpt_dir = os.path.join(config, "ckpt")
        target_urdf_dir = os.path.join(config, "urdf")
        if vlm_ckpt_dir is not None and not os.path.exists(
            target_vlm_ckpt_dir
        ):
            os.symlink(vlm_ckpt_dir, target_vlm_ckpt_dir)

        if urdf_dir is not None and not os.path.exists(target_urdf_dir):
            os.symlink(urdf_dir, target_urdf_dir)

        processor_cfg = load_config_class(
            open(os.path.join(config, f"{processor}.json")).read()
        )

        with in_cwd(config):
            self.processor = processor_cfg()

        self.model = ModelMixin.load_model(
            config, model_prefix="model_0", strict=False
        )
        # self.model = ModelMixin.load_model(config, strict=False)
        self.model.eval()
        self.model.requires_grad_(False)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.take_action_cnt = 0

        self.intrinsic = {
            "left_wrist": np.array(
                [
                    [388.6, 0.0, 240.0, 0.0],
                    [0.0, 388.6, 240.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "right_wrist": np.array(
                [
                    [388.6, 0.0, 240.0, 0.0],
                    [0.0, 388.6, 240.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            "head": np.array(
                [
                    [306.0, 0.0, 360.0, 0.0],
                    [0.0, 306.0, 360.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }

    def data_preprocess(self, obs):
        intrinsic = self.intrinsic
        extrinsic = {}
        for idx, cam in enumerate(utils.ROBOT_CAMERA_NAMES["R1Pro"]):
            cam2base = obs["robot_r1::cam_rel_poses"][7 * idx : 7 * idx + 7]
            pos, quat = cam2base[:3], cam2base[3:]
            rot = utils.quat2mat(quat)  # (3, 3)

            # Add camera coordinate system adjustment:
            # 180 degree rotation around X-axis
            rot_add = utils.euler2mat([np.pi, 0, 0])  # (3, 3)
            rot_matrix = np.matmul(rot, rot_add)  # (3, 3)

            extr = np.eye(4, dtype=float)  # (4, 4)
            extr[:3, :3] = rot_matrix
            # extr[:3, :3] = rot
            extr[:3, 3] = pos
            extrinsic[cam] = extr

        proprio = obs["robot_r1::proprio"]

        base_qvel = proprio[utils.PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]
        base_qvel = base_qvel.reshape(1, -1)

        joint_state = np.hstack(
            [
                # proprio[utils.PROPRIOCEPTION_INDICES['R1Pro']['base_qvel']],
                proprio[utils.PROPRIO_QPOS_INDICES["R1Pro"]["torso"]],
                proprio[utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_arm"]],
                proprio[
                    utils.PROPRIO_QPOS_INDICES["R1Pro"]["left_gripper"]
                ].sum(axis=-1, keepdims=True),
                proprio[utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_arm"]],
                proprio[
                    utils.PROPRIO_QPOS_INDICES["R1Pro"]["right_gripper"]
                ].sum(axis=-1, keepdims=True),
            ]
        )
        joint_state = joint_state.reshape(1, -1)

        eef_state = np.hstack(
            [
                proprio[utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_pos"]],
                proprio[
                    utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_left_quat"]
                ],
                proprio[
                    utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_pos"]
                ],
                proprio[
                    utils.PROPRIOCEPTION_INDICES["R1Pro"]["eef_right_quat"]
                ],
            ]
        )
        eef_state = eef_state.reshape(1, -1)

        # origin image is rgba ==> bgr
        rgb_left_wrist = obs[
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
        ][..., :3][..., ::-1].astype(np.float32)
        rgb_right_wrist = obs[
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
        ][..., :3][..., ::-1].astype(np.float32)
        rgb_head = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3][
            ..., ::-1
        ].astype(np.float32)

        rgb_left_wrist = np.expand_dims(rgb_left_wrist, axis=0) / 255.0
        rgb_right_wrist = np.expand_dims(rgb_right_wrist, axis=0) / 255.0
        rgb_head = np.expand_dims(rgb_head, axis=0) / 255.0
        # print('head img: ')
        # print(rgb_head.shape)
        # print(rgb_head[0, :2, :10, :])
        # print('left img: ')
        # print(rgb_left_wrist[0, :2, :10, :])
        # print('right img: ')
        # print(rgb_right_wrist[0, :2, :10, :])

        # depth
        depth_left_wrist = obs[
            "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"
        ].astype(np.float32)
        depth_right_wrist = obs[
            "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"
        ].astype(np.float32)
        depth_head = obs[
            "robot_r1::robot_r1:zed_link:Camera:0::depth_linear"
        ].astype(np.float32)

        depth_left_wrist = np.nan_to_num(
            depth_left_wrist, nan=10, posinf=10, neginf=0
        )
        depth_right_wrist = np.nan_to_num(
            depth_right_wrist, nan=10, posinf=10, neginf=0
        )
        depth_head = np.nan_to_num(depth_head, nan=10, posinf=10, neginf=0)

        depth_left_wrist = np.expand_dims(depth_left_wrist, axis=0)
        depth_right_wrist = np.expand_dims(depth_right_wrist, axis=0)
        depth_head = np.expand_dims(depth_head, axis=0)

        instruction = obs["instruction"]

        data = MultiArmManipulationInput(
            image={
                "left_wrist": rgb_left_wrist,
                "right_wrist": rgb_right_wrist,
                "head": rgb_head,
            },
            depth={
                "left_wrist": depth_left_wrist,
                "right_wrist": depth_right_wrist,
                "head": depth_head,
            },
            intrinsic=intrinsic,
            t_cam2base=extrinsic,
            history_joint_state=joint_state,
            history_base_qvel=base_qvel,
            mobile_traj=base_qvel,
            instruction=instruction,
        )
        data = self.processor.pre_process(data)

        return data

    def act(self, obs):
        data = self.data_preprocess(obs)
        model_outs = self.model(data)
        actions = self.processor.post_process(data, model_outs).action
        valid_action_step = 32
        # actions = actions[:valid_action_step].cpu().numpy()
        actions = actions[:valid_action_step]

        base = actions[:, :3]
        torso = actions[:, 3:7]
        left_arm = actions[:, 7:14]
        left_gripper = actions[:, 14:15]
        right_arm = actions[:, 15:22]
        right_gripper = actions[:, 22:23]

        actions = torch.cat(
            [base, torso, left_arm, left_gripper, right_arm, right_gripper],
            dim=1,
        )

        print("@" * 100)
        print("action base qvel: ", base[0, :].cpu().numpy())
        print("action torso: ", torso[0, :].cpu().numpy())
        print("action left arm: ", left_arm[0, :].cpu().numpy())
        print("action left gripper: ", left_gripper[0, :].cpu().numpy())
        print("action right arm: ", right_arm[0, :].cpu().numpy())
        print("action right gripper: ", right_gripper[0, :].cpu().numpy())

        return actions

    def reset(self) -> None:
        pass


def main(args: Args) -> None:
    with open(args.policy_config, "r", encoding="utf-8") as fobj:
        config = yaml.safe_load(fobj)
    policy = SEMPolicy(
        config["model_config"],
        config["model_processor"],
        config["vlm_ckpt_dir"],
        config["urdf_dir"],
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={},
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
