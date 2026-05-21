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
import asyncio
import functools
import json
import logging
import os
import socket
import sys
import time
from typing import Any

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_COMMON_ROOT = os.path.dirname(__file__)
if _COMMON_ROOT not in sys.path:
    sys.path.insert(0, _COMMON_ROOT)

import msgpack  # noqa: E402
import numpy as np  # noqa: E402
from holobrain_geniesim3_policy import (  # noqa: E402
    GENIESIM_ACTION_DIM,
    HoloBrainGenieSim3Policy,
    build_policy_from_deploy_config,
)

from robo_orchard_lab.utils import log_basic_config  # noqa: E402

try:
    from websockets.asyncio.server import serve
except ImportError:
    from websockets.server import serve

logger = logging.getLogger(__name__)


def _format_ws_url(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"ws://{host}:{port}"


def _discover_local_ipv4_addresses() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add_address(address: str) -> None:
        if address and address != "0.0.0.0" and address not in seen:
            candidates.append(address)
            seen.add(address)

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_INET,
            type=socket.SOCK_DGRAM,
        ):
            add_address(info[4][0])
    except OSError as exc:
        logger.debug("Failed to resolve hostname for IP hints: %s", exc)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            add_address(sock.getsockname()[0])
    except OSError as exc:
        logger.debug("Failed to inspect default route for IP hints: %s", exc)

    add_address("127.0.0.1")
    candidates.sort(key=lambda item: (item.startswith("127."), item))
    return candidates


def _server_url_hints(host: str, port: int) -> list[str]:
    if host in {"", "0.0.0.0", "::"}:
        return [
            _format_ws_url(address, port)
            for address in _discover_local_ipv4_addresses()
        ]
    return [_format_ws_url(host, port)]


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in (
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


def _unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=_unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class HoloBrainGenieSim3WebsocketServer:
    """Serve HoloBrain GenieSim3 inference over the websocket protocol."""

    def __init__(
        self,
        policy: HoloBrainGenieSim3Policy,
        *,
        host: str,
        port: int,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.request_count = 0

    async def handler(self, websocket, *_) -> None:
        metadata = {
            "server": "holobrain_geniesim3_policy",
            "action_dim": GENIESIM_ACTION_DIM,
            "valid_action_step": self.policy.cfg.valid_action_step,
        }
        await websocket.send(packb(metadata))

        async for message in websocket:
            self.request_count += 1
            if isinstance(message, str):
                logger.warning(
                    "Unexpected text websocket frame on request %s: %s",
                    self.request_count,
                    message,
                )
                continue

            elapsed_ms = 0.0
            error_msg = ""
            try:
                payload = unpackb(message)
                t0 = time.monotonic()
                actions = self.policy.get_actions(payload)
                elapsed_ms = (time.monotonic() - t0) * 1000
            except Exception as exc:
                logger.exception(
                    "Inference error on request %s", self.request_count
                )
                error_msg = f"{type(exc).__name__}: {exc}"
                actions = np.zeros(
                    (
                        self.policy.cfg.valid_action_step,
                        GENIESIM_ACTION_DIM,
                    ),
                    dtype=np.float32,
                )

            response = {
                "actions": actions,
                "model": "holobrain_geniesim3",
                "request_count": self.request_count,
                "error": error_msg,
            }
            await websocket.send(packb(response))
            logger.info(
                "Request %s response sent: %.1fms action_shape=%s",
                self.request_count,
                elapsed_ms,
                tuple(actions.shape),
            )

    async def serve_forever(self) -> None:
        logger.info(
            "Serving GenieSim3 websocket policy at ws://%s:%s",
            self.host,
            self.port,
        )
        logger.info(
            "Connect with one of these websocket URLs: (--infer-host)\n%s",
            "\n".join(
                f"  {url}" for url in _server_url_hints(self.host, self.port)
            ),
        )
        async with serve(
            self.handler,
            self.host,
            self.port,
            max_size=None,
            compression=None,
        ):
            await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve HoloBrain GenieSim3 inference over WebSocket."
    )
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument(
        "--model_processor",
        type=str,
        default="agibot_geniesim3_challenge_processor",
    )
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument(
        "--load_impl",
        type=str,
        default="native",
        choices=["native", "accelerate"],
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--valid_action_step", type=int, default=32)
    parser.add_argument("--sampling_ratio", type=float, default=1.0)
    parser.add_argument("--gripper_limit", type=float, default=1.0)
    parser.add_argument("--use_depth", action=argparse.BooleanOptionalAction)
    parser.set_defaults(use_depth=False)
    return parser.parse_args()


def build_deploy_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_dir": args.model_dir,
        "model_processor": args.model_processor,
        "model_prefix": args.model_prefix,
        "load_impl": args.load_impl,
        "host": args.host,
        "port": args.port,
        "valid_action_step": args.valid_action_step,
        "sampling_ratio": args.sampling_ratio,
        "gripper_limit": args.gripper_limit,
        "use_depth": args.use_depth,
    }


async def main() -> None:
    args = parse_args()
    deploy_cfg = build_deploy_config(args)

    log_basic_config(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(filename)s:%(lineno)d | "
            "%(message)s"
        ),
    )
    logger.info("Deploy config:\n%s", json.dumps(deploy_cfg, indent=4))

    policy = build_policy_from_deploy_config(deploy_cfg)
    server = HoloBrainGenieSim3WebsocketServer(
        policy,
        host=deploy_cfg["host"],
        port=int(deploy_cfg["port"]),
    )
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
