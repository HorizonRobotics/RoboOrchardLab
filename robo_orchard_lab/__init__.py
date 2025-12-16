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

from . import (
    dataset,
    distributed,
    inference,
    models,
    pipeline,
    utils,
)

try:
    from .version import __full_version__, __git_hash__, __version__
except ModuleNotFoundError as _:
    pass


def _set_env():
    import os

    from accelerate.utils import check_cuda_p2p_ib_support

    if not check_cuda_p2p_ib_support():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"


_set_env()
