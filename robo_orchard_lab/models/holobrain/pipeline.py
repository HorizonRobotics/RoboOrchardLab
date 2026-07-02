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

import logging
import os
import shutil
from typing import Any

from robo_orchard_core.utils.config import (
    ClassType_co,
    ConfigInstanceOf,
)

from robo_orchard_lab.inference.basic import (
    InferencePipeline,
    InferencePipelineCfg,
)
from robo_orchard_lab.models.holobrain.processor import (
    HoloBrainProcessor,
    HoloBrainProcessorCfg,
    MultiArmManipulationInput,
    MultiArmManipulationOutput,
)
from robo_orchard_lab.models.mixin import TorchModelMixin
from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    is_empty_directory,
)

logger = logging.getLogger(__file__)


class HoloBrainInferencePipeline(InferencePipeline):
    cfg: "HoloBrainInferencePipelineCfg"
    processor: HoloBrainProcessor

    def __init__(
        self,
        cfg: "HoloBrainInferencePipelineCfg",
        model: TorchModelMixin | None = None,
    ):
        super().__init__(cfg=cfg, model=model)

    def __call__(
        self, data: MultiArmManipulationInput
    ) -> MultiArmManipulationOutput:
        return super().__call__(data)

    def _model_forward(self, data: dict) -> Any:
        return super()._model_forward(data)

    def save_pipeline(
        self,
        directory: str,
        inference_prefix: str = "inference",
        model_prefix: str = "model",
        required_empty: bool = True,
        urdf_dir="./urdf",
        save_model=True,
    ):
        """Saves the full pipeline (model and config) to a directory.

        This method saves the model's weights and configuration by calling its
        `save_model` method, and also saves the pipeline's own configuration
        file.

        Args:
            directory (str): The target directory to save the pipeline to.
            inference_prefix (str): The prefix for the pipeline's config file.
                Defaults to "inference".
            model_prefix (str): The prefix for the model files, passed to the
                model's save method. Defaults to "model".
            required_empty (bool): If True, raises an error if the target
                directory is not empty. Defaults to True.
        """
        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        if save_model:
            self.model.save_model(
                directory=directory,
                model_prefix=model_prefix,
                required_empty=False,
            )
        with open(
            os.path.join(directory, f"{inference_prefix}.config.json"), "w"
        ) as fh:
            cfg_copy = self.cfg.model_copy()
            urdfs = []
            for transform in cfg_copy.processor.transforms:
                if "urdf" in transform:
                    urdf_file = transform["urdf"]
                    urdfs.append(urdf_file)
                    transform["urdf"] = os.path.join(
                        urdf_dir, os.path.basename(urdf_file)
                    )
            if len(urdfs) > 0:
                if not os.path.isabs(urdf_dir):
                    target_urdf_path = os.path.join(directory, urdf_dir)
                else:
                    target_urdf_path = urdf_dir
                os.makedirs(target_urdf_path, exist_ok=True)
                for urdf in urdfs:
                    try:
                        shutil.copy2(urdf, target_urdf_path)
                    except shutil.SameFileError:
                        pass
                    except PermissionError:
                        logger.info(f"copy {urdf} with PermissionError")
            cfg_copy.model_cfg = None  # Avoid redundancy of model config
            fh.write(cfg_copy.model_dump_json(indent=4))


class HoloBrainInferencePipelineCfg(InferencePipelineCfg):
    class_type: ClassType_co[HoloBrainInferencePipeline] = (
        HoloBrainInferencePipeline
    )
    """Class type for the pipeline."""
    processor: ConfigInstanceOf[HoloBrainProcessorCfg] | None = None
    """Processor configuration for the pipeline."""
