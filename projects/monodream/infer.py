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
import json
import os
import random
import re
from datetime import datetime

import cv2

# navigation
import imageio
import numpy as np
import PIL
import PIL.Image
from habitat import Env
from habitat.core.agent import Agent
from habitat.datasets import make_dataset
from habitat.utils.visualizations import maps
from tqdm import trange
from VLN_CE.vlnce_baselines.config.default import get_config

from robo_orchard_lab.inference import InferencePipelineMixin
from robo_orchard_lab.models.monodream.processor import (
    MonoDreamInput,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num", type=int, required=True, help="chunks of evluation"
    )
    parser.add_argument(
        "--split-id", type=int, required=True, help="chunks ID of evluation"
    )
    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results",
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    config = get_config(args.exp_config)
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    dataset_split = dataset.get_splits(args.split_num)[args.split_id]

    # time for intermedia store
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.result_path, "tmp")
    new_dir = os.path.join(base_dir, current_time)
    os.makedirs(new_dir, exist_ok=True)
    new_ep_dir = os.path.join(new_dir, str(args.split_id))
    os.makedirs(new_ep_dir, exist_ok=True)
    new_dir = new_ep_dir
    os.makedirs(os.path.join(args.result_path, "log"), exist_ok=True)
    os.makedirs(os.path.join(args.result_path, "video"), exist_ok=True)

    print("config::")
    print(config.TASK_CONFIG)

    env = Env(config=config.TASK_CONFIG, dataset=dataset_split)

    if args.visualize:
        is_require_map = True
    else:
        is_require_map = False
    agent = NavAgent(args, is_require_map)
    num_episodes = len(env.episodes)

    early_rotation_steps = config.EVAL.EARLY_STOP_ROTATION
    early_stop_steps = config.EVAL.EARLY_STOP_STEPS

    target_key = {
        "distance_to_goal",
        "success",
        "spl",
        "path_length",
        "oracle_success",
    }

    count = 0

    for _ in trange(
        num_episodes,
        desc=config.EVAL.IDENTIFICATION + "-{}".format(args.split_id),
    ):
        obs = env.reset()
        iter_step = 0
        agent.reset()

        continuse_rotation_count = 0
        last_dtg = 999

        save_file_path = os.path.join(
            os.path.join(args.result_path, "log"),
            "stats_{}.json".format(env.current_episode.episode_id),
        )
        if os.path.isfile(save_file_path):
            continue

        while not env.episode_over:
            info = env.get_metrics()

            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            img_save_dir = os.path.join(new_dir, "{}.png".format(iter_step))

            action = agent.act(
                obs, info, env.current_episode.episode_id, img_save_dir
            )

            if (
                continuse_rotation_count > early_rotation_steps
                or iter_step > early_stop_steps
            ):
                action = {"action": 0}

            iter_step += 1
            obs = env.step(action)

        action_step = agent.action_step
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        result_dict["step"] = action_step
        count += 1

        with open(
            os.path.join(
                os.path.join(args.result_path, "log"),
                "stats_{}.json".format(env.current_episode.episode_id),
            ),
            "w",
        ) as f:
            try:
                json.dump(result_dict, f, indent=4)
            except Exception as e:
                print(f"Error saving file: {e}")

            t_num = 0
            while not os.path.exists(
                os.path.join(
                    os.path.join(args.result_path, "log"),
                    "stats_{}.json".format(env.current_episode.episode_id),
                )
            ):
                json.dump(result_dict, f, indent=4)
                t_num += 1
                if t_num >= 5:
                    print(
                        "file save bad!!!",
                        "stats_{}.json".format(env.current_episode.episode_id),
                    )
                    break


class NavAgent(Agent):
    def __init__(self, args, require_map=True):
        print("Initialize NavAgent")

        self.result_path = args.result_path
        self.require_map = require_map

        os.makedirs(self.result_path, exist_ok=True)

        # load inference pipeline
        self.monodream_pipline = InferencePipelineMixin.load(args.model_path)
        self.monodream_pipline.model.init_components(args.model_path)

        print("Initialization Complete")

        self.rgb_list = []
        self.topdown_map_list = []

        self.count_id = 0
        self.action_done_list = []
        self.action_step = 0
        self.reset()

    def reset(self):
        if self.require_map:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(
                    self.result_path, "video", "{}.mp4".format(self.episode_id)
                )
                imageio.mimsave(output_video_path, self.topdown_map_list)

        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []
        self.action_done_list = []
        self.action_step = 0
        self.first_forward = False

    def extract_result(self, output):
        # id: 0-stop, 1 move forward, 2 turn left, 3 turn right

        output = output.split(",")[0]
        if "stop" in output or "Stop" in output:
            return 0, None
        elif "left" in output or "Left" in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return None, None
            match = match.group()
            return 2, float(match)
        elif "right" in output or "Right" in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return None, None
            match = match.group()
            return 3, float(match)
        elif "forward" in output or "Forward" in output:
            match = re.search(r"-?\d+", output)
            if match is None:
                return None, None
            match = match.group()
            return 1, float(match)
        return 0, None

    def predict_inference(self, prompt):
        if len(self.rgb_list) > 0:
            if len(self.rgb_list) > 9:
                step = (len(self.rgb_list) - 1) / 8
                indices = [int(round(j * step)) for j in range(9)]
                current_image_list = [self.rgb_list[j] for j in indices]
            else:
                current_image_list = self.rgb_list

        data = MonoDreamInput(
            image_paths=current_image_list,
            instruction=prompt,
        )
        response = self.monodream_pipline(data).text

        return response

    def addtext(self, image, instuction, navigation):
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instuction, font, 0.5, 2)[0]
        text_y = h + (50 + textsize[1]) // 2

        y_line = text_y + 0 * textsize[1]

        words = instuction.split(" ")
        x = 10
        line = ""

        for word in words:
            test_line = line + " " + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(
                    new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2
                )
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(
            new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2
        )

        return new_image

    def act(self, observations, info, episode_id, img_save_dir):
        self.episode_id = episode_id
        rgb = observations["rgb"]

        # intermedia store
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb_rgb)
        self.rgb_list.append(pil_image)

        if self.require_map:
            tdm = info["top_down_map_vlnce"]
            tdm_map = tdm["map"]
            for unwanted in [7, 8, 10, 12, 13, 14]:
                tdm_map[tdm_map == unwanted] = 1
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb_rgb, top_down_map), axis=1)

        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)

            if self.require_map:
                img = self.addtext(
                    output_im,
                    observations["instruction"]["text"],
                    "Pending action: {}".format(temp_action),
                )
                self.topdown_map_list.append(img)
            return {"action": temp_action}

        navigation = self.predict_inference(
            observations["instruction"]["text"]
        )

        if self.require_map:
            img = self.addtext(
                output_im, observations["instruction"]["text"], navigation
            )
            self.topdown_map_list.append(img)

        action_index, num = self.extract_result(navigation[:-1])

        if action_index == 0:
            self.pending_action_list.append(0)
        elif action_index == 1:
            for _ in range(min(3, int(num / 25))):
                self.pending_action_list.append(1)

        elif action_index == 2:
            for _ in range(min(3, int(num / 15))):
                self.pending_action_list.append(2)

        elif action_index == 3:
            for _ in range(min(3, int(num / 15))):
                self.pending_action_list.append(3)

        if action_index is None or len(self.pending_action_list) == 0:
            self.pending_action_list.append(random.randint(1, 3))

        self.action_step += 1
        return {"action": self.pending_action_list.pop(0)}


if __name__ == "__main__":
    main()
