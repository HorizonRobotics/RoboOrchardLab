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

from __future__ import annotations
import argparse
import json
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

_COMMON_ROOT = Path(__file__).resolve().parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))

_REPO_ROOT = _COMMON_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402
from gevent.pywsgi import WSGIServer  # noqa: E402
from holobrain_utils import download_file  # noqa: E402

from robo_orchard_lab.models.holobrain.processor import (  # noqa: E402
    HoloBrainProcessor,
    MultiArmManipulationInput,
)
from robo_orchard_lab.models.mixin import ModelMixin  # noqa: E402

logger = logging.getLogger(__name__)

# This script currently supports dual-arm real-world deployment only.
# Extend request decoding and action formatting for other embodiments.
MODEL_WEIGHT_FILE = "model.safetensors"
MODEL_CONFIG_FILE = "model.config.json"
REQUIRED_REQUEST_KEYS = (
    "left_color",
    "middle_color",
    "right_color",
    "left_depth",
    "middle_depth",
    "right_depth",
    "left_intrinsic",
    "middle_intrinsic",
    "right_intrinsic",
    "left_arm_state",
    "right_arm_state",
    "instruction",
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HoloBrain Model Server",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6050,
        help="Port to run the server on.",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="holobrain",
        help="Route name for the inference endpoint.",
    )
    parser.add_argument(
        "--num_joints_per_arm",
        type=int,
        default=7,
        help="Number of joints per arm.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("model"),
        help="Local model directory.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        help="Remote checkpoint URL used to fill missing model files.",
    )
    parser.add_argument(
        "--model_processor",
        type=str,
        default=None,
        help=(
            "Processor json name, with or without .json. Required for "
            "--model_url downloads unless model_dir already contains a single "
            "processor json."
        ),
    )
    parser.add_argument(
        "--vlm_ckpt_dir",
        type=Path,
        default=None,
        help="Optional VLM checkpoint directory to link as model_dir/ckpt.",
    )
    parser.add_argument(
        "--urdf_dir",
        type=Path,
        default=None,
        help="Optional URDF directory to link as model_dir/urdf.",
    )
    parser.add_argument("--clip_action_len", "-c", type=int, default=None)
    parser.add_argument("--delay_horizon", "-d", type=int, default=None)
    parser.add_argument(
        "--interpolation",
        type=positive_float,
        default=200 / 30,
        help="Interpolation ratio between model and real-world action rates.",
    )
    parser.add_argument(
        "--max_action_delta",
        type=positive_float,
        default=None,
        help=(
            "Optional max per-dimension action change between adjacent output "
            "steps. Extra steps are linearly inserted when exceeded."
        ),
    )
    return parser.parse_args(argv)


def configure_logging(server_name: str, port: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.name = f"{server_name} server in [{port}]"


def _load_npy_field(request_data: dict[str, Any], key: str) -> np.ndarray:
    return np.load(BytesIO(request_data[key].read()))


def _decode_remaining_actions(
    request_data: dict[str, Any],
    interpolation: float,
) -> np.ndarray | None:
    if request_data.get("remaining_actions") is None:
        return None

    remaining_actions = _load_npy_field(
        request_data, "remaining_actions"
    ).astype(np.float32)[None]
    if remaining_actions.size == 0:
        return None

    remaining_actions = (
        torch.nn.functional.interpolate(
            torch.from_numpy(remaining_actions).permute(0, 2, 1),
            scale_factor=max(
                1 / interpolation, 1 / remaining_actions.shape[1]
            ),
            mode="linear",
            align_corners=True,
        )
        .permute(0, 2, 1)
        .numpy()
    )
    return remaining_actions[:, 1:]


def normalize_processor_file(model_processor: str) -> str:
    if model_processor.endswith(".json"):
        return model_processor
    return f"{model_processor}.json"


def resolve_model_processor(
    model_dir: Path,
    model_processor: str | None,
) -> str:
    if model_processor is not None:
        return normalize_processor_file(model_processor)

    processor_files = sorted(
        path.name
        for path in model_dir.glob("*.json")
        if path.name != MODEL_CONFIG_FILE
    )
    if len(processor_files) == 1:
        return processor_files[0]
    if not processor_files:
        raise ValueError(
            "--model_processor is required because no processor json was "
            f"found in {model_dir}."
        )
    raise ValueError(
        "--model_processor is required because multiple processor json files "
        f"were found in {model_dir}: {processor_files}."
    )


def download_model_if_needed(
    model_url: str | None,
    model_dir: Path,
    processor_file: str,
    vlm_ckpt_dir: Path | None,
    urdf_dir: Path | None,
) -> None:
    """Download artifacts when the local model directory is incomplete."""
    required_model_files = (
        MODEL_WEIGHT_FILE,
        MODEL_CONFIG_FILE,
        processor_file,
    )
    if all(
        (model_dir / file_name).exists() for file_name in required_model_files
    ):
        link_model_resources(model_dir, vlm_ckpt_dir, urdf_dir)
        logger.info(
            "Model files already present in %s, skipping download.", model_dir
        )
        return

    if model_url is None:
        raise FileNotFoundError(
            f"Model files not found in '{model_dir}' and --model_url was not "
            "provided."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_url = model_url.rstrip("/")
    urls = [
        (
            f"{model_url}/{MODEL_WEIGHT_FILE}",
            model_dir / MODEL_WEIGHT_FILE,
        ),
        (
            f"{model_url}/{MODEL_CONFIG_FILE}",
            model_dir / MODEL_CONFIG_FILE,
        ),
        (
            "/".join(model_url.split("/")[:-2] + [processor_file]),
            model_dir / processor_file,
        ),
    ]

    for url, dest in urls:
        if dest.exists():
            logger.info("%s already exists, skipping.", dest.name)
            continue

        logger.info("Downloading %s -> %s", url, dest)
        download_file(url, str(dest))

    link_model_resources(model_dir, vlm_ckpt_dir, urdf_dir)


def link_model_resources(
    model_dir: Path,
    vlm_ckpt_dir: Path | None,
    urdf_dir: Path | None,
) -> None:
    link_model_resource(model_dir / "ckpt", vlm_ckpt_dir)
    link_model_resource(model_dir / "urdf", urdf_dir)


def link_model_resource(link_path: Path, source_dir: Path | None) -> None:
    if source_dir is not None and not link_path.exists():
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(source_dir)


class HoloBrainPolicy:
    """Load a HoloBrain model and run real-world inference requests."""

    def __init__(
        self,
        *,
        model_dir: Path,
        model_url: str | None = None,
        processor_file: str,
        vlm_ckpt_dir: Path | None = None,
        urdf_dir: Path | None = None,
        num_joints_per_arm: int,
        clip_action_len: int | None = None,
        delay_horizon: int | None = None,
        interpolation: float = 200 / 30,
        max_action_delta: float | None = None,
    ) -> None:
        download_model_if_needed(
            model_url,
            model_dir,
            processor_file,
            vlm_ckpt_dir,
            urdf_dir,
        )
        self.model = ModelMixin.load_model(str(model_dir), load_impl="native")

        from robo_orchard_lab.models.rtc_plugin.rtc_plugin import (
            RTCInferencePlugin,
        )

        if self.model.decoder.async_inference_plugin is None:
            self.model.decoder.async_inference_plugin = RTCInferencePlugin()

        self.processor = HoloBrainProcessor.load(
            str(model_dir),
            processor_file,
        )
        self.model.eval()
        self.model.cuda()
        self.model.requires_grad_(False)
        self.model.decoder.num_test_traj = 1
        self.interpolation = interpolation
        self.num_joints_per_arm = num_joints_per_arm
        self.clip_action_len = clip_action_len
        self.delay_horizon = delay_horizon
        self.max_action_delta = max_action_delta
        if self.interpolation <= 0:
            raise ValueError("interpolation must be positive.")
        if self.max_action_delta is not None and self.max_action_delta <= 0:
            raise ValueError("max_action_delta must be positive.")
        logger.info("Model initialized successfully, model_dir: %s", model_dir)

    def data_preprocess(self, request_data: dict[str, Any]) -> dict[str, Any]:
        images = {
            "left": [
                _load_npy_field(request_data, "left_color").astype(np.uint8)
            ],
            "right": [
                _load_npy_field(request_data, "right_color").astype(np.uint8)
            ],
            "middle": [
                _load_npy_field(request_data, "middle_color").astype(np.uint8)
            ],
        }

        depths = {
            "left": [
                _load_npy_field(request_data, "left_depth").astype(np.float64)
                / 1000.0
            ],
            "right": [
                _load_npy_field(request_data, "right_depth").astype(np.float64)
                / 1000.0
            ],
            "middle": [
                _load_npy_field(request_data, "middle_depth").astype(
                    np.float64
                )
                / 1000.0
            ],
        }

        left_arm_state = _load_npy_field(
            request_data, "left_arm_state"
        ).astype(np.float32)
        right_arm_state = _load_npy_field(
            request_data, "right_arm_state"
        ).astype(np.float32)
        joint_state = np.concatenate(
            [left_arm_state, right_arm_state],
            axis=-1,
        )[None, :]

        intrinsic_values = np.eye(4)[None].repeat(3, axis=0)
        intrinsic_values[0, :3] = _load_npy_field(
            request_data, "left_intrinsic"
        ).astype(np.float64)
        intrinsic_values[1, :3] = _load_npy_field(
            request_data, "right_intrinsic"
        ).astype(np.float64)
        intrinsic_values[2, :3] = _load_npy_field(
            request_data, "middle_intrinsic"
        ).astype(np.float64)
        intrinsics = {
            "left": intrinsic_values[0],
            "right": intrinsic_values[1],
            "middle": intrinsic_values[2],
        }

        instruction = request_data.get("instruction", "")
        logger.info("Received instruction: %s", instruction)

        model_input = MultiArmManipulationInput(
            image=images,
            depth=depths,
            history_joint_state=joint_state,
            intrinsic=intrinsics,
            instruction=instruction,
            remaining_actions=_decode_remaining_actions(
                request_data,
                self.interpolation,
            ),
            delay_horizon=int(request_data.get("delay_horizon", 0)),
        )
        return self.processor.pre_process(model_input)

    def get_action(self, request_data: dict[str, Any]) -> dict[str, Any]:
        pre_processed_input = self.data_preprocess(request_data)
        model_output = self.model(pre_processed_input)
        return self.data_postprocess(pre_processed_input, model_output)

    def data_postprocess(
        self,
        pre_processed_input: dict[str, Any],
        model_output: Any,
    ) -> dict[str, Any]:
        actions = self.processor.post_process(
            model_output,
            pre_processed_input,
        ).action

        actions = self._clip_actions_if_needed(actions, pre_processed_input)
        actions = torch.nn.functional.interpolate(
            actions.permute(1, 0)[None],
            scale_factor=self.interpolation,
            mode="linear",
            align_corners=True,
        )[0].permute(1, 0)
        actions = self._limit_action_delta(actions)

        result = {
            "left_arm_actions": actions[:, : self.num_joints_per_arm]
            .cpu()
            .numpy()
            .tolist(),
            "right_arm_actions": actions[:, self.num_joints_per_arm :]
            .cpu()
            .numpy()
            .tolist(),
            "action_horizon": len(actions),
        }
        if self.delay_horizon is not None:
            result["action_horizon"] = min(
                self.delay_horizon,
                result["action_horizon"],
            )

        logger.info(
            "Inference succeeded, action_shape=%s", tuple(actions.shape)
        )
        return result

    def _limit_action_delta(self, actions: torch.Tensor) -> torch.Tensor:
        if self.max_action_delta is None or actions.shape[0] <= 1:
            return actions

        action_array = actions.detach().cpu().numpy()
        start_actions = action_array[:-1]
        end_actions = action_array[1:]
        action_delta = end_actions - start_actions
        max_delta = np.max(np.abs(action_delta), axis=1)
        segment_counts = np.maximum(
            np.ceil(max_delta / self.max_action_delta).astype(np.int64),
            1,
        )

        total_steps = int(segment_counts.sum()) + 1
        limited_array = np.empty(
            (total_steps, action_array.shape[1]),
            action_array.dtype,
        )
        limited_array[0] = action_array[0]

        segment_ids = np.repeat(
            np.arange(len(segment_counts), dtype=np.int64),
            segment_counts,
        )
        step_ids = (
            np.arange(len(segment_ids), dtype=np.int64)
            - np.repeat(
                np.r_[0, np.cumsum(segment_counts[:-1])],
                segment_counts,
            )
            + 1
        )
        alpha = step_ids[:, None] / segment_counts[segment_ids, None]
        limited_array[1:] = start_actions[segment_ids] + action_delta[
            segment_ids
        ] * alpha.astype(action_array.dtype)

        return torch.from_numpy(limited_array).to(
            device=actions.device,
            dtype=actions.dtype,
        )

    def _clip_actions_if_needed(
        self,
        actions: torch.Tensor,
        pre_processed_input: dict[str, Any],
    ) -> torch.Tensor:
        if self.clip_action_len is None:
            return actions

        current_joint = (
            pre_processed_input["hist_robot_state"][0, -1:, :, 0].cpu().numpy()
        )
        distance = np.abs(actions.cpu().numpy() - current_joint)
        gripper_indices = (
            self.num_joints_per_arm - 1,
            self.num_joints_per_arm * 2 - 1,
        )
        if gripper_indices[-1] >= distance.shape[-1]:
            raise ValueError(
                "num_joints_per_arm is incompatible with action dimension: "
                f"num_joints_per_arm={self.num_joints_per_arm}, "
                f"action_dim={distance.shape[-1]}."
            )
        static_mask = np.logical_or(
            np.any(distance[..., gripper_indices] > 0.02, axis=-1),
            np.any(distance > 0.1, axis=-1),
        )
        if not np.any(static_mask):
            valid_action_step = actions.shape[0]
        else:
            valid_action_step = static_mask.argmax()
        return actions[: max(self.clip_action_len, valid_action_step)]


def create_app(
    policy: HoloBrainPolicy,
    server_name: str,
) -> Flask:
    app = Flask(__name__)

    @app.route(f"/{server_name}", methods=["POST"])
    def model_infer() -> Response | tuple[Response, int]:
        data = {**request.files, **request.form}
        for key in REQUIRED_REQUEST_KEYS:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400
        try:
            result = policy.get_action(data)
        except Exception as exc:
            logger.exception("Error during inference")
            return jsonify({"error": str(exc)}), 500

        return Response(json.dumps(result), mimetype="application/json")

    return app


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(args.server_name, args.port)

    processor_file = resolve_model_processor(
        args.model_dir,
        args.model_processor,
    )
    policy = HoloBrainPolicy(
        model_dir=args.model_dir,
        model_url=args.model_url,
        processor_file=processor_file,
        vlm_ckpt_dir=args.vlm_ckpt_dir,
        urdf_dir=args.urdf_dir,
        num_joints_per_arm=args.num_joints_per_arm,
        clip_action_len=args.clip_action_len,
        delay_horizon=args.delay_horizon,
        interpolation=args.interpolation,
        max_action_delta=args.max_action_delta,
    )
    app = create_app(policy, args.server_name)

    logger.info(
        "Serving HoloBrain model at /%s on port %s",
        args.server_name,
        args.port,
    )
    http_server = WSGIServer(("", args.port), app)
    http_server.serve_forever()


if __name__ == "__main__":
    main()
