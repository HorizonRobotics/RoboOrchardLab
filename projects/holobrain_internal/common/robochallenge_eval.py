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
import logging
import os
import sys

from holobrain_robochallenge_policy import (
    HoloBrainPolicy,
    job_loop,
    run_local_client_loop,
)

logger = logging.getLogger(__file__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HoloBrain RoboChallenge online inference",
    )
    parser.add_argument("--mock", action="store_true")

    parser.add_argument("--user_token", type=str, default=None)
    parser.add_argument("--submission_id", type=str, default=None)

    parser.add_argument("--instruction", type=str, default="")

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--model_processor", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument("--clip_action_len", type=int, default=32)
    parser.add_argument("--interpolation", type=float, default=1.0)
    parser.add_argument("--vlm_ckpt_dir", type=str, default=None)
    parser.add_argument("--urdf_dir", type=str, default=None)
    parser.add_argument("--visualize_output_file", type=str, default=None)

    parser.add_argument("--image_size", type=str, default="308,252")
    parser.add_argument("--duration", type=float, default=0.05)

    def int_or_none(x):
        if x in ("None", "none", "null"):
            return None
        return int(x)

    parser.add_argument("--delay_horizon", type=int_or_none, default=None)
    parser.add_argument("--rtc_max_horizon", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rc_repo = os.environ["ROBOCHALLENGE_INFERENCE_REPO"]
    if rc_repo not in sys.path:
        sys.path.insert(0, rc_repo)

    from robot.interface_client import InterfaceClient

    policy = HoloBrainPolicy(
        config=args.model_config,
        processor=args.model_processor,
        model_prefix=args.model_prefix,
        vlm_ckpt_dir=args.vlm_ckpt_dir,
        urdf_dir=args.urdf_dir,
        clip_action_len=args.clip_action_len,
        interpolation=args.interpolation,
        visualize_output_file=args.visualize_output_file,
        delay_horizon=args.delay_horizon,
        rtc_max_horizon=args.rtc_max_horizon,
    )

    embodiment = policy.embodiment
    image_type = policy.processor.cfg.cam_names
    if embodiment in {"ur5", "arx5"}:
        action_type = "leftjoint"
    else:
        action_type = "joint"
    image_size = [int(x) for x in args.image_size.split(",")]
    assert len(image_size) == 2
    duration = args.duration
    logger.info(
        f"embodiment: {embodiment}, image_type: {image_type}, "
        f"action_type: {action_type}, image_size: {image_size} "
        f"duration: {duration}"
    )

    if not args.mock:
        if not args.user_token:
            raise ValueError("--user_token is required when not using --mock")
        if not args.submission_id:
            raise ValueError(
                "--submission_id is required when not using --mock"
            )
        client = InterfaceClient(args.user_token)
        job_loop(
            client,
            policy,
            args.submission_id,
            image_size,
            image_type,
            action_type,
            duration,
        )
    else:
        client = InterfaceClient("test_user", mock=True)
        client.update_job_info("test_job", "test_robot")
        run_local_client_loop(
            client,
            policy,
            image_size,
            image_type,
            action_type,
            duration,
            instruction=args.instruction,
        )


if __name__ == "__main__":
    main()
