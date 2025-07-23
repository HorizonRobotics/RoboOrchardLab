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

import importlib
import logging
import os

import requests
import torch
from safetensors.torch import load_model

logger = logging.getLogger(__file__)


def load_config(config_file):
    assert config_file.endswith(".py")
    module_name = os.path.split(config_file)[-1][:-3]
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


class GetFile:
    def __init__(self, url):
        self.url = url

    def __enter__(self):
        if not self.url.startswith("http"):
            return self.url
        file_name = "_" + self.url.split("/")[-1]
        with requests.get(self.url, stream=True) as r:
            r.raise_for_status()
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        self.url = file_name
        return file_name

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def load_checkpoint(model, checkpoint=None, accelerator=None, **kwargs):
    if checkpoint is None:
        return

    logger.info(f"load checkpoint: {checkpoint}")
    with GetFile(checkpoint) as checkpoint:
        if checkpoint.endswith(".safetensors"):
            missing_keys, unexpected_keys = load_model(
                model, checkpoint, strict=False, **kwargs
            )
        else:
            state_dict = torch.load(checkpoint, weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False, **kwargs
            )
        if accelerator is None or accelerator.is_main_process:
            logger.info(
                f"num of missing_keys: {len(missing_keys)},"
                f"num of unexpected_keys: {len(unexpected_keys)}"
            )
            logger.info(
                f"missing_keys:\n {missing_keys}\n"
                f"unexpected_keys:\n {unexpected_keys}"
            )
