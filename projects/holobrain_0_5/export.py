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


"""Export a deployable HoloBrain 0.5 policy from a Python config.

Writes the same layout as ``train.py`` so the evaluation entries accept the
result as ``--model-dir``, and reloads every artifact to prove it. Use this to
produce an export without training, or to rebuild one from a checkpoint.

Run from the repository root:
    python -m projects.holobrain_0_5.export --workspace ./export \\
        --kwargs '{"checkpoint": "./ckpt/action.safetensors"}'
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from projects.holobrain_0_5.utils import (
    load_checkpoint,
    load_config,
    parse_kwargs,
)
from robo_orchard_lab.models.holobrain_0_5 import HoloBrainProcessor
from robo_orchard_lab.models.holobrain_0_5.pipeline import (
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.utils import log_basic_config

__all__ = ["export"]

logger = logging.getLogger(__name__)


def export(config_path, overrides, workspace):
    """Export model, processors and inference pipelines, then reload them."""
    module = load_config(config_path)
    config = dict(module.config)
    config.update(overrides)
    output = Path(workspace).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger.info("\n" + json.dumps(config, indent=4))
    shutil.copytree(
        Path(config_path).resolve().parent,
        output / "configs",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (output / "training_config.json").write_text(json.dumps(config, indent=2))

    model = module.build_model(config)
    load_checkpoint(model, config.get("checkpoint"))
    model.save_model(str(output), required_empty=False)
    logger.info("Exported model.safetensors")

    processors = module.build_processors(config)
    for name, processor in processors.items():
        processor.save(str(output), f"{name}_processor.json")
        pipeline = HoloBrainInferencePipeline(
            HoloBrainInferencePipelineCfg(processor=processor.cfg), model
        )
        pipeline.save_pipeline(
            str(output),
            inference_prefix=f"{name}_inference",
            required_empty=False,
            save_model=False,
        )
        logger.info("Exported %s_processor.json", name)

    del model
    ModelMixin.load_model(str(output), load_impl="native")
    logger.info("Reloaded the exported model")
    for name in processors:
        HoloBrainProcessor.load(str(output), f"{name}_processor.json")
        HoloBrainInferencePipeline.load_pipeline(
            directory=str(output),
            inference_prefix=f"{name}_inference",
            load_weights=False,
            load_impl="native",
        )
        logger.info("Reloaded the %s processor and pipeline", name)
    logger.info("Export ready: %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs/config_holobrain_0_5.py"),
    )
    parser.add_argument("--kwargs")
    parser.add_argument("--workspace", default="./export")
    args = parser.parse_args()
    log_basic_config(
        format="%rank %(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )
    export(args.config, parse_kwargs(args.kwargs), args.workspace)
