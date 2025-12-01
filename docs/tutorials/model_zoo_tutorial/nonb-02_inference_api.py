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

# ruff: noqa: E501 D415 D205 E402

"""Creating, Saving, and Loading Inference Pipelines
=======================================================================

This tutorial demonstrates how to build, save, and load an end-to-end
inference pipeline.
"""

# sphinx_gallery_thumbnail_path = '_static/images/sphx_glr_install_thumb.png'

# %%
# Core Concepts
# ---------------------------------------------------------
# The Inference API builds on the Model Zoo API (:py:class:`~robo_orchard_lab.models.torch_model.TorchModelMixin`)
# and adds components for data processing. The key classes are:
#
# 1. :py:class:`~robo_orchard_lab.inference.processor.mixin.ProcessorMixin`: The base class for defining
#    data pre-processing (e.g., from NumPy to Tensor) and post-processing
#    (e.g., from Tensor to NumPy).
#
# 2. :py:class:`~robo_orchard_lab.inference.mixin.InferencePipelineMixin`: The base class for the
#    inference pipeline. It holds the model and processor.
#
# 3. :py:class:`~robo_orchard_lab.inference.basic.InferencePipeline`: A concrete implementation
#    of the pipeline that orchestrates ``pre_process``, ``collate_fn``,
#    ``model.forward``, and ``post_process``.
#
# 4. :py:class:`~robo_orchard_lab.inference.basic.InferencePipelineCfg`: The configuration
#    class that defines the entire pipeline, including the nested model
#    config, processor config, and collate function.
#

# %%
# Step 1: Define All Components (Model, Processor)
# ---------------------------------------------------------
# A pipeline requires a model and a processor. We will define minimal
# versions of each, following the same pattern as the Model Zoo tutorial.
#
import os
import shutil
import subprocess
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from robo_orchard_lab.inference import (
    InferencePipelineCfg,
    InferencePipelineMixin,
)
from robo_orchard_lab.inference.processor import (
    ProcessorMixin,
    ProcessorMixinCfg,
)
from robo_orchard_lab.models import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
)


# 1. Define the Model
class SimpleNet(ModelMixin):
    def __init__(self, cfg: "SimpleNetCfg"):
        super().__init__(cfg)
        self.fc = nn.Linear(cfg.input_size, cfg.output_size)

    def forward(self, batch: Dict[str, torch.Tensor]):
        # The model expects a dictionary from the collate function
        return self.fc(batch["data"])


class SimpleNetCfg(TorchModuleCfg[SimpleNet]):
    class_type: ClassType_co[SimpleNet] = SimpleNet
    input_size: int = 10
    output_size: int = 2


# 2. Define the Processor
class SimpleProcessor(ProcessorMixin):
    def __init__(self, cfg: "SimpleProcessorCfg"):
        super().__init__(cfg)
        self.scale = cfg.scale_factor

    def pre_process(self, data: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convert raw NumPy array to a tensor dictionary."""
        tensor_data = torch.from_numpy(data).float() * self.scale
        return {"data": tensor_data}

    def post_process(
        self, model_outputs: torch.Tensor, model_input: Any = None
    ) -> np.ndarray:
        """Convert output tensor back to a NumPy array."""
        return model_outputs.cpu().detach().numpy()


class SimpleProcessorCfg(ProcessorMixinCfg[SimpleProcessor]):
    class_type: ClassType_co[SimpleProcessor] = SimpleProcessor
    scale_factor: float = 1.0


# 3. Define the Collate Function
def simple_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Stack the pre-processed data into a batch."""
    stacked_data = torch.stack([item["data"] for item in batch], dim=0)
    return {"data": stacked_data}


print("Model, Processor, and Collate Function defined.")

# %%
# Step 2: Define the Pipeline Configuration
# ---------------------------------------------------------
# Now we assemble all components into a single
# :py:class:`~robo_orchard_lab.inference.pipeline.InferencePipelineCfg`.
#
# We use :py:class:`~robo_orchard_core.utils.config.ConfigInstanceOf`
# to nest the model and processor configurations.
#

# 1. Create instances of the component configs
model_cfg = SimpleNetCfg(input_size=10, output_size=2)
processor_cfg = SimpleProcessorCfg(scale_factor=0.5)

# 2. Create the main pipeline config
pipeline_config = InferencePipelineCfg(
    # Nest the model config
    model_cfg=model_cfg,
    # Nest the processor config
    processor=processor_cfg,
    # Assign the collate function
    collate_fn=simple_collate_fn,
    # Set batch size for batch inference
    batch_size=8,
)

print("Pipeline configuration created.")

# %%
# Step 3: Instantiate and Save the Pipeline
# ---------------------------------------------------------
# Instantiating the config will create the ``InferencePipeline``, which
# in turn creates the ``SimpleNet`` and ``SimpleProcessor`` instances.
#
# The :py:meth:`~robo_orchard_lab.inference.mixin.InferencePipelineMixin.save`
# method will save:
#
# 1.  The pipeline config (``inference.config.json``).
# 2.  The model config (``model.config.json``).
# 3.  The model weights (``model.safetensors``).
#


# 1. Instantiate the pipeline by calling the config object
pipeline = pipeline_config()
print("Pipeline created:", pipeline)

# 2. Call the .save() method
output_dir = ".workspace/pipeline_checkpoint"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

pipeline.save(output_dir)

print(f"Pipeline has been saved to the `{output_dir}` directory.")
print(subprocess.check_output(["tree", output_dir]).decode())

# %%
# Step 4: Load and Run the Pipeline
# ---------------------------------------------------------
# We use :py:meth:`~robo_orchard_lab.inference.mixin.InferencePipelineMixin.load`
# to load the entire pipeline.
#
# This single call reconstructs the pipeline, model, and processor,
# and loads the model weights.
#

# 1. Load the pipeline
loaded_pipeline = InferencePipelineMixin.load(output_dir)
loaded_pipeline.model.eval()  # Set model to eval mode
print("Pipeline loaded:", loaded_pipeline)

# 2. Run inference with raw data
# We provide a raw NumPy array, just as defined in the
# processor's pre_process method.
raw_data = np.arange(10, dtype=np.float32)

# The pipeline handles:
# pre_process -> collate_fn -> model.forward -> post_process
output = loaded_pipeline(raw_data)

print(f"\nRaw input data:\n{raw_data}")
print(f"Pipeline output (post-processed):\n{output}")

# 3. Run batch inference
raw_batch = [
    np.arange(10, dtype=np.float32),
    np.arange(10, 20, dtype=np.float32),
]
batch_output = list(loaded_pipeline(raw_batch))
print("\nBatch input (2 items).")
print(f"Batch output (1 batch of 2 items):\n{batch_output[0]}")
assert batch_output[0].shape == (2, 2)
