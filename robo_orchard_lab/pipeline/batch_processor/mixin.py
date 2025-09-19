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

from abc import ABC, abstractmethod
from typing import Callable

from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHookArgs,
    PipelineHooks,
)

__all__ = ["BatchProcessorMixin"]


class BatchProcessorMixin(ABC):
    """A processor for handling batches in a training or inference pipeline."""

    @abstractmethod
    def __call__(
        self,
        pipeline_hooks: PipelineHooks,
        on_batch_hook_args: PipelineHookArgs,
        model: Callable,
    ) -> None:
        """Executes the batch processing pipeline.

        The processed outputs and loss (if applicable) are stored in the
        `on_batch_hook_args` workspace.

        Args:
            pipeline_hooks (PipelineHooks): The pipeline hooks to be triggered
                during batch processing.
            on_batch_hook_args (PipelineHookArgs): The workspace for the
                on_batch hook. It should contain the following
                attributes:
                  - accelerator: The Accelerator instance.
                  - batch: The batch of data to be processed.
                After the call, it will contain:
                  - reduce_loss: The computed loss.
                  - model_outputs: The model outputs.
            model (Callable): The model function or callable.
        """
        pass
