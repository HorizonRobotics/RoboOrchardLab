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


import os

import cv2
import numpy as np

# 每个 action step (25 Hz) 取到的帧，攒下来随下一次 get_action 一起送出去。
#
# 为什么需要这个：deploy.py 本来就每步都调 update_obs，但
# `client_server/ws/model_client.py:60-62` 把它吞在本地
# (`self._latest_obs = obs; return None`)，只有 get_action 才真的发包。于是
# policy 的帧缓冲每次**前向**才进一帧，而不是每个 action step 一帧 ——
# 一条 episode 只有 25(swap_T)~100(imitate) 次前向，episode 开头几次前向的
# 历史比训练时短得多。
#
# 这里不改传输层：把攒下的帧挂在紧邻 get_action 的那次 obs 上，
# msgpack_numpy 会照常序列化（observation 是 dict[str, Any]，不是 forbid-extra
# 的 Frame）。前向次数不变。
_STREAM = os.environ.get("HOLOBRAIN_STREAM_HISTORY", "0").lower() in (
    "1",
    "true",
    "yes",
)
_STREAM_CAMS = [
    x
    for x in os.environ.get("HOLOBRAIN_STREAM_HISTORY_CAMS", "cam_head").split(",")
    if x
]


def _stream_wh():
    raw = os.environ.get("HOLOBRAIN_STREAM_HISTORY_WH", "352x256")
    w, _, h = raw.partition("x")
    return int(w), int(h)


def _thumbs(obs):
    """只取 history 用得到的那一路相机，并先降到缓冲分辨率再发。

    发原图会把每次 get_action 的包撑到几十 MB；降到 352x256 之后 15 帧约 4 MB，
    与本来就在发的 3 路主图同一量级。policy 侧的缓冲反正也要 resize 到这个尺寸。
    """
    w, h = _stream_wh()
    vision = obs.get("vision") or {}
    out = {}
    for cam in _STREAM_CAMS:
        data = vision.get(cam)
        if not isinstance(data, dict) or "color" not in data:
            continue
        out[cam] = cv2.resize(np.asarray(data["color"]), (w, h))
    return out


def eval_one_episode(TASK_ENV, model_client):  # noqa: N803
    model_client.call(func_name="reset")

    pending = []
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        if _STREAM:
            obs = dict(obs)
            obs["history_frames"] = pending
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        pending = []

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            next_obs = TASK_ENV.get_obs()
            if _STREAM:
                pending.append(_thumbs(next_obs))
            model_client.call(func_name="update_obs", obs=next_obs)


def eval_one_episode_batch(TASK_ENV, model_client):  # noqa: N803
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(
            func_name="get_action_batch",
            obs=env_idx_list,
        )

        chunk_size = len(actions[0])
        for action_idx in range(chunk_size):
            current_actions = [
                env_actions[action_idx] for env_actions in actions
            ]
            TASK_ENV.take_action_batch(current_actions, env_idx_list)
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active = [
                index
                for index, env_idx in enumerate(env_idx_list)
                if env_idx in running
            ]
            actions = [actions[index] for index in active]
            env_idx_list = [env_idx_list[index] for index in active]
            model_client.call(
                func_name="update_obs_batch",
                obs=TASK_ENV.get_obs_batch(env_idx_list),
            )
