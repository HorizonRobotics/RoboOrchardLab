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
    ProcessorMixinCfgType_co,
)


class ComposeProcessor(ProcessorMixin):
    """A processor that chains multiple processors together.

    This processor acts as a container to apply a sequence of other processors
    serially. It allows for building complex data processing pipelines from
    smaller, modular components.
    """

    cfg: "ComposeProcessorCfg"  # for type hint

    def __init__(self, cfg: "ComposeProcessorCfg"):
        super().__init__(cfg)
        self.processors: list[ProcessorMixin] = [
            cfg_i() for cfg_i in self.cfg.processors
        ]

    def pre_process(self, data):
        """Applies the `pre_process` method of each processor in sequence.

        The output of one processor becomes the input to the next.

        Args:
            data: The initial raw input data.

        Returns:
            The data after being transformed by all processors in the sequence.
        """
        for ts in self.processors:
            data = ts.pre_process(data)
        return data

    def post_process(self, model_outputs, model_input):
        # Apply post-processing in reverse order of pre-processing
        for ts in reversed(self.processors):
            model_outputs = ts.post_process(model_outputs, model_input)
        return model_outputs


class ComposeProcessorCfg(ProcessorMixinCfg[ComposeProcessor]):
    class_type: ClassType_co[ComposeProcessor] = ComposeProcessor
    processors: list[ProcessorMixinCfgType_co]  # type: ignore
