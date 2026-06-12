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

import atexit
import logging
import os

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from holobrain_utils import download_file

from robo_orchard_lab.dataset.horizon_manipulation import (
    HorizonManipulationLmdbDataset,
)
from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
    AddItems,
    ItemSelection,
)
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.models.rtc_plugin.rtc_plugin import (
    RTCInferencePlugin,
)

logger = logging.getLogger(__file__)

TABLE30V2_DEFAULT_SOURCE_SIZES = {
    "aloha": [640, 480],
    "arx5": [1280, 720],
    "dos_w1": [640, 480],
    "ur5": [640, 480],
}


def download_job_ckpt_processor(
    ckpt_url, processor_name, output_dir="./model", model_prefix="model"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(output_dir, 0o755)

    while ckpt_url.endswith("/"):
        ckpt_url = ckpt_url[:-1]
    model_url = f"{ckpt_url}/model.safetensors"
    model_config_url = f"{ckpt_url}/{model_prefix}.config.json"
    processor_url = "/".join(
        ckpt_url.split("/")[:-2] + [f"{processor_name}.json"]
    )
    print(
        f"model_ckpt: {model_url}\n"
        f"model_config: {model_config_url}\n"
        f"procssor: {processor_url}"
    )
    for url in [model_url, model_config_url, processor_url]:
        file_name = os.path.join(output_dir, url.split("/")[-1])
        if url.endswith("config.json"):
            file_name = file_name.replace(
                f"{model_prefix}.config.json", "model.config.json"
            )
        download_file(url, file_name)


class FFmpegVideoWriter:
    def __init__(
        self,
        output_path: str,
        frame_size: tuple[int, int],
        fps: int = 10,
    ):
        self._generator = imageio_ffmpeg.write_frames(
            str(output_path),
            frame_size,
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="warning",
            output_params=["-movflags", "+faststart"],
        )
        self._generator.send(None)
        atexit.register(self.release)

    def write(self, frame: np.ndarray) -> None:
        self._generator.send(np.ascontiguousarray(frame, dtype=np.uint8))

    def release(self) -> None:
        self._generator.close()


class HoloBrainPolicy:
    def __init__(
        self,
        config,
        processor: str = None,
        model_prefix="model",
        vlm_ckpt_dir=None,
        urdf_dir=None,
        clip_action_len: int | None = None,
        interpolation: float = 1.0,
        visualize_output_file=False,
        delay_horizon=None,
        rtc_max_horizon=16,
    ):
        if processor is None:
            processor = "processor"
        if config.startswith("http"):
            download_job_ckpt_processor(
                ckpt_url=config,
                processor_name=processor,
                output_dir="./holobrain_eval_model",
                model_prefix=model_prefix,
            )
            config = "./holobrain_eval_model"
        logger.info(f"model config: {config}, processor: {processor}")

        target_vlm_ckpt_dir = os.path.join(config, "ckpt")
        target_urdf_dir = os.path.join(config, "urdf")
        if vlm_ckpt_dir is not None and not os.path.exists(
            target_vlm_ckpt_dir
        ):
            os.symlink(vlm_ckpt_dir, target_vlm_ckpt_dir)
        if urdf_dir is not None and not os.path.exists(target_urdf_dir):
            os.symlink(urdf_dir, target_urdf_dir)

        self.processor = HoloBrainProcessor.load(config, f"{processor}.json")
        self.processor.cfg.load_depth = False
        self.processor.struction_to_dict.load_depth = False
        self.model = ModelMixin.load_model(config, load_impl="native")
        self.delay_horizon = delay_horizon
        if (
            hasattr(self.model.decoder, "async_inference_plugin")
            and self.model.decoder.async_inference_plugin is None
        ):
            self.model.decoder.async_inference_plugin = RTCInferencePlugin(
                max_horizon=rtc_max_horizon
            )
            for transform in self.processor.transforms:
                if isinstance(transform, ItemSelection):
                    transform.keys.extend(
                        ["delay_horizon", "remaining_actions"]
                    )

        self.model.eval()
        self.model.requires_grad_(False)
        # self.model.decoder.enable_torch_compile()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.base_intrinsic = None
        for transform in self.processor.transforms:
            if (
                isinstance(transform, AddItems)
                and transform.items.get("intrinsic") is not None
            ):
                self.base_intrinsic = transform.items["intrinsic"].copy()
                break
        assert self.base_intrinsic is not None

        self.embodiment = None
        for embodiment in TABLE30V2_DEFAULT_SOURCE_SIZES.keys():
            if embodiment in processor:
                assert self.embodiment is None
                self.embodiment = embodiment
        assert self.embodiment is not None
        logger.info(f"embodiment: {self.embodiment}")
        self.interpolation = interpolation
        self.clip_action_len = clip_action_len

        self.visualize_output_file = visualize_output_file
        self.video_writer = None
        self.last_job_id = None
        self.reset()

    def data_preprocess(self, state, prompt, delay_horizon):
        images = {}
        joint_state = np.asarray(state["action"], dtype=np.float32)

        for i, cam_name in enumerate(self.processor.cfg.cam_names):
            img_buffer = np.frombuffer(
                state["images"][cam_name], dtype=np.uint8
            )
            bgr_image = cv2.imdecode(img_buffer, cv2.IMREAD_ANYCOLOR)
            if bgr_image is None:
                raise ValueError(f"Failed to decode image for {cam_name}")
            images[cam_name] = [bgr_image]

            source_width, source_height = TABLE30V2_DEFAULT_SOURCE_SIZES[
                self.embodiment
            ]
            scale_x = float(bgr_image.shape[1]) / float(source_width)
            scale_y = float(bgr_image.shape[0]) / float(source_height)

            trans_matrix = np.eye(4, dtype=np.float64)
            trans_matrix[0, 0] = scale_x
            trans_matrix[1, 1] = scale_y

            for transform in self.processor.transforms:
                if (
                    isinstance(transform, AddItems)
                    and transform.items.get("intrinsic") is not None
                ):
                    transform.items["intrinsic"][i] = (
                        trans_matrix @ self.base_intrinsic[i]
                    )
                    break

        model_input = MultiArmManipulationInput(
            image=images,
            depth=None,
            intrinsic=None,
            history_joint_state=joint_state[None],
            instruction=prompt,
        )
        if delay_horizon is not None and self.remaining_actions is not None:
            if len(self.remaining_actions) > 0:
                model_input.remaining_actions = self.remaining_actions[None]
                model_input.delay_horizon = delay_horizon
                logger.info(
                    f"remaining_actions: {model_input.remaining_actions.shape}"
                )
        return self.processor.pre_process(model_input)

    def reset(self):
        self.current_step = 0
        self.current_progress = 0
        self.remaining_actions = None

    def infer(self, state, prompt="") -> np.ndarray:
        job_id = state["job_id"]

        if self.last_job_id is None or job_id != self.last_job_id:
            self.reset()

        self.last_job_id = job_id

        delay_horizon = self.delay_horizon
        interpolation = self.interpolation
        clip_action_len = self.clip_action_len
        data = self.data_preprocess(state, prompt, delay_horizon)
        model_outs = self.model(data)

        actions = self.processor.post_process(data, model_outs).action
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()

        if self.embodiment == "dos_w1":
            actions[..., [6, 13]] -= 0.002

        prepared = np.asarray(actions, dtype=np.float32)
        if clip_action_len is not None:
            current_joint = (
                data["hist_robot_state"][0, -1:, :, 0].cpu().numpy()
            )
            distance = np.abs(prepared - current_joint)
            static_mask = np.any(distance > 0.02, axis=-1)
            if not np.any(static_mask):
                valid_action_step = prepared.shape[0]
            else:
                valid_action_step = static_mask.argmax()

            self.remaining_actions = prepared[
                max(clip_action_len, valid_action_step) :
            ]
            prepared = prepared[: max(clip_action_len, valid_action_step)]
        else:
            self.remaining_actions = None
        self.current_step += prepared.shape[0]

        if interpolation and interpolation != 1 and len(prepared):
            prepared = np.concatenate(
                [np.array(state["action"])[None], prepared]
            )
            target_len = max(int(len(prepared) * interpolation) - 1, 1)
            source_index = np.arange(len(prepared), dtype=np.float32)
            target_index = np.linspace(
                0,
                len(prepared) - 1,
                num=target_len,
                dtype=np.float32,
            )
            upsampled = np.empty(
                (target_len, prepared.shape[1]), dtype=np.float32
            )
            for column_index in range(prepared.shape[1]):
                upsampled[:, column_index] = np.interp(
                    target_index,
                    source_index,
                    prepared[:, column_index],
                )
            prepared = upsampled

        logger.info(
            f"prompt: {prompt}, action length: {len(prepared)}, "
            f"job_id: {job_id}."
        )
        if self.visualize_output_file is not None:
            self.vis_callback(data, prepared)
        return prepared.tolist()

    def vis_callback(self, data, actions):
        imgs = data["imgs"][0]
        projection_mat = data["projection_mat"][0]
        current_robot_state = data["hist_robot_state"][0]
        embodiedment_mat = data["embodiedment_mat"][0]

        future_robot_state = data["kinematics"][0].joint_state_to_robot_state(
            torch.Tensor(actions).to(current_robot_state),
            embodiedment_mat=embodiedment_mat,
        )
        all_robot_state = torch.cat([current_robot_state, future_robot_state])
        all_robot_state = all_robot_state.flatten(0, 1)

        vis_img = HorizonManipulationLmdbDataset.get_vis_imgs(
            imgs, projection_mat, all_robot_state, channel_conversion=True
        )

        if self.video_writer is None:
            self.video_writer = FFmpegVideoWriter(
                self.visualize_output_file,
                frame_size=(vis_img.shape[1], vis_img.shape[0]),
            )
        self.video_writer.write(vis_img)
