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
import logging
import os
import shutil

from holobrain_utils import load_checkpoint, load_config

from robo_orchard_lab.models.holobrain import HoloBrainProcessor
from robo_orchard_lab.models.holobrain.pipeline import (
    HoloBrainInferencePipeline,
    HoloBrainInferencePipelineCfg,
)
from robo_orchard_lab.models.mixin import ModelMixin
from robo_orchard_lab.utils import log_basic_config

logger = logging.getLogger(__file__)

#: Tensor count of a fully built MemoryVLAMemory (perceptual + cognitive),
#: measured on every on-state run of the port since 09. A package that
#: reloads with fewer has lost part of the module.
MEMORYVLA_TENSORS = 68


def _assert_memory_survived_export(
    workspace, model_path, processors, required
):
    """A package exported with the memory on must still have it.

    Everything else about this port fails loudly by now; this was the one
    remaining way to end up switched on and computing nothing. `export.py`
    assembles the artefacts that evaluation actually loads, and nothing
    between here and a benchmark run asks whether the memory made it: a
    model whose `memoryvla` came back None, or a processor whose
    ItemSelection dropped `step_index`, evaluates cleanly and just scores
    worse. Checked here because this is the last place that still knows the
    config said the memory was on.
    """
    reloaded = ModelMixin.load_model(model_path, load_impl="native")
    memory = getattr(reloaded, "memoryvla", None)
    if memory is None:
        raise RuntimeError(
            "memoryvla.enable=True, but the model reloaded from "
            f"{model_path} has memoryvla=None. The exported package would "
            "evaluate as a plain baseline and report nothing about it."
        )
    n = len(list(memory.state_dict()))
    if n != MEMORYVLA_TENSORS:
        raise RuntimeError(
            f"memoryvla reloaded with {n} tensors, expected "
            f"{MEMORYVLA_TENSORS}. Part of the module did not survive the "
            "round trip through the package."
        )

    for dataset_name in processors:
        if required is not None and dataset_name not in required:
            continue
        name = f"{dataset_name}_processor.json"
        with open(os.path.join(workspace, name)) as f:
            spec = json.load(f)
        text = json.dumps(spec)
        if '"step_index"' not in text:
            raise RuntimeError(
                f"{name} has no `step_index` among its ItemSelection keys. "
                "The memory reads it to place each frame in time; without it "
                "every retrieval at evaluation raises, or -- worse, if some "
                "future default fills it in -- silently shares one position "
                "across the episode. It is added only when the memory is on "
                "(config_robodojo_dataset.py:288-293), so its absence means "
                "the processor was built from a config that had it off."
            )
    logger.info(
        "memoryvla survived export: %d tensors, step_index in every "
        "exported processor.",
        n,
    )


def main(args):
    os.makedirs(args.workspace, exist_ok=True)
    shutil.copytree(
        "configs",
        os.path.join(args.workspace, "configs"),
        dirs_exist_ok=True,
    )

    config = load_config(args.config)
    build_model = config.build_model
    build_processors = config.build_processors
    config = config.config

    if args.kwargs is not None:
        if os.path.isfile(args.kwargs):
            with open(args.kwargs, "r") as f:
                kwargs = json.load(f)
        else:
            kwargs = json.loads(args.kwargs)
        config.update(kwargs)
    logger.info("\n" + json.dumps(config, indent=4))

    required_datasets = args.dataset_names
    if required_datasets is not None:
        required_datasets = required_datasets.split(",")

    # export data processors and reload test
    processors = build_processors(config)
    for dataset_name, processor in processors.items():
        if (
            required_datasets is not None
            and dataset_name not in required_datasets
        ):
            continue
        processor_name = f"{dataset_name}_processor.json"
        processor.save(args.workspace, processor_name)
        logger.info(f"Export {processor_name} successfully.")
        _processor = HoloBrainProcessor.load(args.workspace, processor_name)
        logger.info(f"Reload {processor_name} successfully.")

    # # export model and reload test
    model = build_model(config)
    load_checkpoint(model, config.get("checkpoint"))
    model_path = os.path.join(args.workspace, "model")
    model.save_model(model_path, required_empty=False)
    logger.info("Export model successfully.")
    if args.reload_test:
        _model = ModelMixin.load_model(model_path, load_impl="native")
        logger.info("Reload model successfully.")

    # export inference.config.json for each dataset's pipeline
    for dataset_name, processor in processors.items():
        if (
            required_datasets is not None
            and dataset_name not in required_datasets
        ):
            continue
        inference_cfg = HoloBrainInferencePipelineCfg(
            class_type=HoloBrainInferencePipeline,
            model_cfg=None,
            processor=processor.cfg,
        )
        pipeline = HoloBrainInferencePipeline(inference_cfg, model)
        pipeline.save_pipeline(
            model_path,
            inference_prefix=f"{dataset_name}_inference",
            required_empty=False,
            save_model=False,
        )
        logger.info(f"Export {dataset_name} inference pipeline successfully.")

        if args.reload_test:
            HoloBrainInferencePipeline.load_pipeline(
                model_path,
                inference_prefix=f"{dataset_name}_inference",
                load_impl="native",
            )
            logger.info(
                f"Reload {dataset_name} inference pipeline successfully."
            )

    # Unconditional, not gated on --reload_test: the packages that go to a
    # benchmark are the ones nobody remembered to pass the flag for.
    if (config.get("memoryvla") or {}).get("enable", False):
        _assert_memory_survived_export(
            args.workspace, model_path, processors, required_datasets
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--workspace", type=str, default="./workspace")
    parser.add_argument("--reload_test", action="store_true")
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()
    log_basic_config(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d | %(message)s",  # noqa: E501
        level=logging.INFO,
    )

    logger.info(f"Export to workspace dir {args.workspace}")
    main(args)
