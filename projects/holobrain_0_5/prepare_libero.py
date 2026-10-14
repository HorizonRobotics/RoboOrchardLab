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


"""Replay standard LIBERO HDF5 demonstrations into RGB-D action LMDBs."""

import argparse
import json
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import h5py
import numpy as np

from projects.holobrain_0_5.libero_runtime import (
    depthimg2meters,
    prepare_libero_runtime,
)
from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import BaseIndexData
from robo_orchard_lab.dataset.lmdb.lmdb_wrapper import Lmdb
from robo_orchard_lab.utils.filesystem import rename_noreplace

__all__ = ["pack_libero", "main"]


def pack_libero(input_dir, output_dir, libero_root, max_episodes=None):
    """Write RGB, metric depth, camera poses and absolute controller targets.

    Each frame is restored from its recorded MuJoCo state before rendering.
    Original delta actions are converted by the simulator's OSC controller,
    preserving its scaling and rotation conventions. Input files must be the
    standard LIBERO HDF5 layout with states, actions and model_file metadata.
    The destination must not already exist; incomplete outputs are not usable.
    """
    prepare_libero_runtime(libero_root)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )
    from robosuite.utils.transform_utils import mat2quat, quat2axisangle

    paths = sorted(Path(input_dir).glob("*_demo.hdf5"))
    if not paths:
        raise ValueError(f"No LIBERO demonstration files in {input_dir}")
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ExitStack() as stack:
        temporary = stack.enter_context(
            TemporaryDirectory(prefix=".libero-", dir=destination.parent)
        )
        output = Path(temporary) / "dataset"
        output.mkdir()
        stores = {}
        for name in ["index", "meta", "image", "depth"]:
            store = Lmdb(str(output / name), writable=True, commit_step=100)
            stack.callback(store.close)
            stores[name] = store
        for path in paths:
            if max_episodes is not None and count >= max_episodes:
                break
            task_name = path.stem.removesuffix("_demo")
            bddl = (
                Path(get_libero_path("bddl_files"))
                / path.parent.name
                / f"{task_name}.bddl"
            )
            with h5py.File(path, "r") as file:
                data = file["data"]
                metadata = json.loads(
                    data.attrs.get(
                        "env_args", data.attrs.get("env_info", "{}")
                    )
                )
                kwargs = metadata.get("env_kwargs", metadata)
                if isinstance(kwargs, str):
                    kwargs = json.loads(kwargs)
                kwargs = dict(kwargs)
                controller_config = kwargs.pop("controller_configs", None)
                if controller_config is not None:
                    kwargs["controller"] = controller_config["type"]
                kwargs.update(
                    bddl_file_name=str(bddl),
                    camera_names=["agentview", "robot0_eye_in_hand"],
                    camera_heights=256,
                    camera_widths=256,
                    camera_depths=True,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                )
                env = OffScreenRenderEnv(**kwargs)
                try:
                    for demo_name in sorted(
                        data.keys(), key=lambda x: int(x.split("_")[-1])
                    ):
                        if max_episodes is not None and count >= max_episodes:
                            break
                        demo = data[demo_name]
                        states, actions = (
                            demo["states"][()],
                            demo["actions"][()],
                        )
                        if (
                            len(states) != len(actions)
                            or actions.shape[1] != 7
                        ):
                            raise ValueError(
                                "Invalid state/action alignment in "
                                f"{path}:{demo_name}"
                            )
                        env.reset()
                        xml = demo.attrs["model_file"]
                        if isinstance(xml, bytes):
                            xml = xml.decode()
                        xml_tree = ET.fromstring(postprocess_model_xml(xml))
                        # Demonstrations contain paths from the recording host.
                        # Robosuite paths were resolved above; relocate LIBERO
                        # meshes/textures using their stable assets suffix.
                        for element in xml_tree.iter():
                            filename = element.get("file")
                            if filename and not Path(filename).is_file():
                                suffix = filename.split("/assets/", 1)
                                if len(suffix) != 2:
                                    raise FileNotFoundError(filename)
                                asset = (
                                    Path(get_libero_path("assets")) / suffix[1]
                                )
                                if not asset.is_file():
                                    raise FileNotFoundError(asset)
                                element.set("file", str(asset))
                        env.reset_from_xml_string(
                            ET.tostring(xml_tree, encoding="unicode")
                        )
                        env.sim.reset()
                        if controller_config is not None:
                            # Recreate the recorded OSC settings before replay.
                            for robot in env.env.robots:
                                robot.controller_config.update(
                                    controller_config
                                )
                                robot._load_controller()
                        uuid = f"{path.parent.name}/{task_name}/{demo_name}"
                        cameras = {
                            "agentview": "agentview",
                            "eye_in_hand": "robot0_eye_in_hand",
                        }
                        extrinsic = {key: [] for key in cameras}
                        intrinsic = {key: [] for key in cameras}
                        ee_states, gripper_states, absolute_actions = (
                            [],
                            [],
                            [],
                        )
                        for step, (state, action) in enumerate(
                            zip(states, actions, strict=True)
                        ):
                            obs = env.set_init_state(state)
                            ee_states.append(
                                np.concatenate(
                                    [
                                        obs["robot0_eef_pos"],
                                        quat2axisangle(obs["robot0_eef_quat"]),
                                    ]
                                )
                            )
                            gripper_states.append(
                                obs["robot0_gripper_qpos"].copy()
                            )
                            for key, camera in cameras.items():
                                rgb = obs[f"{camera}_image"]
                                # Preserve render orientation for the reader.
                                ok, encoded = cv2.imencode(".png", rgb)
                                if not ok:
                                    raise RuntimeError(
                                        "Failed to encode LIBERO image"
                                    )
                                stores["image"].write(
                                    f"{uuid}/{key}/{step}", encoded.tobytes()
                                )
                                depth = depthimg2meters(
                                    env, obs[f"{camera}_depth"]
                                ).astype(np.float32)
                                stores["depth"].write(
                                    f"{uuid}/{key}/{step}",
                                    depth.reshape(256, 256, 1),
                                )
                                extrinsic[key].append(
                                    get_camera_extrinsic_matrix(
                                        env.sim, camera
                                    )
                                )
                                intrinsic[key].append(
                                    get_camera_intrinsic_matrix(
                                        env.sim, camera, 256, 256
                                    )
                                )
                            controller = env.env.robots[0].controller
                            controller.use_delta = True
                            controller.update(force=True)
                            controller.set_goal(action[:6])
                            absolute_actions.append(
                                np.concatenate(
                                    [
                                        controller.goal_pos,
                                        quat2axisangle(
                                            mat2quat(controller.goal_ori)
                                        ),
                                        action[6:],
                                    ]
                                )
                            )
                        meta = stores["meta"]
                        values = dict(
                            camera_names=list(cameras),
                            extrinsic={
                                k: np.stack(v) for k, v in extrinsic.items()
                            },
                            intrinsic={
                                k: np.stack(v) for k, v in intrinsic.items()
                            },
                            action=np.stack(absolute_actions),
                            instructions=env.language_instruction,
                        )
                        values["observation/ee_state"] = np.stack(ee_states)
                        values["observation/gripper_state"] = np.stack(
                            gripper_states
                        )
                        for key, value in values.items():
                            meta.write(f"{uuid}/{key}", value)
                        stores["index"].write(
                            count,
                            BaseIndexData(
                                uuid=uuid,
                                num_steps=len(actions),
                                task_name=task_name,
                                simulation=True,
                            ).model_dump(),
                        )
                        count += 1
                finally:
                    env.close()
        stores["index"].write("__len__", count, commit=True)
        for store in stores.values():
            store.close()
        rename_noreplace(output, destination)
    return count


def main():
    """Pack a LIBERO suite with an optional episode limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    episodes = pack_libero(
        args.input, args.output, args.libero_root, args.max_episodes
    )
    print(f"Packed {episodes} episodes")


if __name__ == "__main__":
    main()
