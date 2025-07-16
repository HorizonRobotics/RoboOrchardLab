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

from robo_orchard_lab.inference.processor.mixin import (
    ClassType_co,
    ProcessorMixin,
    ProcessorMixinCfg,
)


class IdentityProcessor(ProcessorMixin):
    """A processor that performs no operations.

    This processor serves as a pass-through component, returning the data it
    receives without any modification. It is useful as a default processor or
    as a placeholder in pipelines where no pre-processing or post-processing
    is required.
    """

    cfg: "IdentityProcessorCfg"  # for type hint

    def __init__(self, cfg: "IdentityProcessorCfg"):
        super().__init__(cfg)

    def pre_process(self, data):
        """Returns the input data without modification.

        Args:
            data: The raw input data.

        Returns:
            The same input data, unchanged.
        """
        return data

    def post_process(self, batch, model_outputs):
        """Returns the model outputs without modification."""
        return model_outputs


class IdentityProcessorCfg(ProcessorMixinCfg[IdentityProcessor]):
    class_type: ClassType_co[IdentityProcessor] = IdentityProcessor
